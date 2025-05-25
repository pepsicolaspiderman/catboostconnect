from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

app = FastAPI(title="Сервис прогнозирования трат по категориям")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['dow'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    # лаги нескольких порядков
    for lag in [1,2,3,7,14,30]:
        df[f'lag_{lag}'] = df.groupby('category')['amount'].shift(lag).fillna(0)
    # скользящие суммы и средние
    df['roll_7_sum'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
    df['roll_7_mean'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df['roll_30_sum'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).sum())
    df['roll_30_mean'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    # дни с последней ненулевой транзакцией
    df['last_nz'] = df['date'].where(df['amount']>0)
    df['last_nz'] = df.groupby('category')['last_nz'].ffill()
    df['days_since'] = (df['date'] - df['last_nz']).dt.days.fillna(999).astype(int)
    return df

# Pydantic-модели
class Transaction(BaseModel):
    date: str        # "YYYY-MM-DD"
    category: str
    amount: float

class ForecastRequest(BaseModel):
    transactions: List[Transaction]
    forecast_days: int = Field(..., gt=0, description="Число дней для прогноза")

class Prediction(BaseModel):
    date: str
    category: str
    value: float

class ForecastResponse(BaseModel):
    metrics: Dict[str, float]
    predictions: List[Prediction]

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # 1. Подготовка данных
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="Список транзакций пуст")
    df['date'] = pd.to_datetime(df['date'])

    # 2. Ресемплинг по дням для каждой категории
    cats = df['category'].unique()
    frames = []
    for cat in cats:
        tmp = df[df['category']==cat].set_index('date')['amount'].resample('D').sum().to_frame()
        tmp['category'] = cat
        tmp['amount'] = tmp['amount'].fillna(0)
        tmp = tmp.reset_index()
        frames.append(tmp)
    hist = pd.concat(frames, ignore_index=True).sort_values(['category','date']).reset_index(drop=True)

    # 3. Генерация признаков
    hist = build_features(hist)
    feats = ['category','day','month','dow','is_weekend'] + \
            [f'lag_{lag}' for lag in [1,2,3,7,14,30]] + \
            ['roll_7_sum','roll_7_mean','roll_30_sum','roll_30_mean','days_since']
    # Признак наличия траты
    hist['target_bin'] = (hist['amount']>0).astype(int)

    X = hist[feats].copy()
    X['category'] = X['category'].astype('category')
    y_bin = hist['target_bin']
    y_amt = hist.loc[hist['target_bin']==1,'amount']

    # 4. Обучение моделей
    clf = CatBoostClassifier(iterations=150, learning_rate=0.1, depth=6,
                             random_seed=42, verbose=False, allow_writing_files=False)
    clf.fit(X, y_bin, cat_features=['category'])
    reg = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6,
                            random_seed=42, verbose=False, allow_writing_files=False)
    reg.fit(X[y_bin==1], y_amt, cat_features=['category'])

    # Метрики in-sample
    prob = clf.predict_proba(X)[:,1]
    pred_bin = clf.predict(X)
    pred_amt = reg.predict(X)
    pred = prob * pred_amt
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
    acc = accuracy_score(y_bin, pred_bin)
    auc = roc_auc_score(y_bin, prob)
    rmse = mean_squared_error(hist['amount'], pred, squared=False)

    # 5. Прогноз на будущее
    last_date = hist['date'].max()
    history = hist.copy()
    future = []
    for i in range(1, req.forecast_days+1):
        d = last_date + pd.Timedelta(days=i)
        for cat in cats:
            sub = history[history['category']==cat]
            # собрать фичи
            row = {
                'category': cat,
                'day': d.day, 'month': d.month,
                'dow': d.dayofweek, 'is_weekend': int(d.dayofweek in [5,6])
            }
            for lag in [1,2,3,7,14,30]:
                row[f'lag_{lag}'] = sub['amount'].iloc[-lag] if len(sub)>=lag else 0
            row['roll_7_sum'] = sub['amount'].shift(1).rolling(7, min_periods=1).sum().iloc[-1]
            row['roll_7_mean'] = sub['amount'].shift(1).rolling(7, min_periods=1).mean().iloc[-1]
            row['roll_30_sum'] = sub['amount'].shift(1).rolling(30, min_periods=1).sum().iloc[-1]
            row['roll_30_mean'] = sub['amount'].shift(1).rolling(30, min_periods=1).mean().iloc[-1]
            row['days_since'] = int((d - sub[sub['amount']>0]['date'].max()).days) \
                              if not sub[sub['amount']>0].empty else 999
            Xf = pd.DataFrame([row])[feats]
            Xf['category'] = Xf['category'].astype('category')
            p = clf.predict_proba(Xf)[0,1]
            r = reg.predict(Xf)[0]
            val = max(0.0, float(p * r))
            history = pd.concat([history, pd.DataFrame([{**row,'amount':val,'date':d}])], ignore_index=True)
            future.append({'date': d.strftime('%Y-%m-%d'), 'category': cat, 'value': round(val,2)})

    return ForecastResponse(
        metrics={'acc': round(acc,3), 'auc': round(auc,3), 'rmse': round(rmse,2)},
        predictions=future
    )

if __name__=='__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=80)
