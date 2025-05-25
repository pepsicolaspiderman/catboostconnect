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
    # Лаги различных порядков
    for lag in [1,2,3,7,14,30]:
        df[f'lag_{lag}'] = df.groupby('category')['amount'].shift(lag).fillna(0)
    # Скользящие суммы и средние
    df['roll_7_sum'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
    df['roll_7_mean'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df['roll_30_sum'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).sum())
    df['roll_30_mean'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    # Количество дней с тратами
    df['roll_7_count'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).apply(lambda y: (y>0).sum(), raw=True))
    df['roll_30_count'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).apply(lambda y: (y>0).sum(), raw=True))
    # Перцентиль (75%)
    df['roll_30_p75'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).quantile(0.75))
    # Временные признаки: фурье для недельного и месячного циклов
    import numpy as np
    df['sin_dow'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['sin_dom'] = np.sin(2 * np.pi * (df['day'] - 1) / 30)
    df['cos_dom'] = np.cos(2 * np.pi * (df['day'] - 1) / 30)
    # Дни с последней ненулевой транзакцией
    df['last_nz'] = df['date'].where(df['amount']>0)
    df['last_nz'] = df.groupby('category')['last_nz'].ffill()
    df['days_since'] = (df['date'] - df['last_nz']).dt.days.fillna(999).astype(int)
    return df

class Transaction(BaseModel):
    date: str
    category: str
    amount: float

class ForecastRequest(BaseModel):
    transactions: List[Transaction]
    forecast_days: int = Field(..., gt=0)

class Prediction(BaseModel):
    date: str
    category: str
    value: float

class ForecastResponse(BaseModel):
    predictions: List[Prediction]

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="Список транзакций пуст")
    df['date'] = pd.to_datetime(df['date'])
    cats = df['category'].unique()
    frames = []
    for cat in cats:
        tmp = df[df['category']==cat].set_index('date')['amount'].resample('D').sum().to_frame()
        tmp['category'] = cat
        tmp['amount'] = tmp['amount'].fillna(0)
        tmp = tmp.reset_index()
        frames.append(tmp)
    hist = pd.concat(frames, ignore_index=True).sort_values(['category','date']).reset_index(drop=True)
    hist = build_features(hist)
    feats = [
        'category','day','month','dow','is_weekend',
        'sin_dow','cos_dow','sin_dom','cos_dom',
        ] + [f'lag_{lag}' for lag in [1,2,3,7,14,30]] + [
        'roll_7_sum','roll_7_mean','roll_30_sum','roll_30_mean',
        'roll_7_count','roll_30_count','roll_30_p75','days_since'
    ]
    hist['target_bin'] = (hist['amount']>0).astype(int)
    X = hist[feats].copy()
    X['category'] = X['category'].astype('category')
    y_bin = hist['target_bin']
    y_amt = hist.loc[hist['target_bin']==1,'amount']

    clf = CatBoostClassifier(iterations=150, learning_rate=0.1, depth=6, random_seed=42, verbose=False, allow_writing_files=False)
    clf.fit(X, y_bin, cat_features=['category'])
    reg = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, random_seed=42, verbose=False, allow_writing_files=False)
    reg.fit(X[y_bin==1], y_amt, cat_features=['category'])

    last_date = hist['date'].max()
    history = hist.copy()
    future = []
    for i in range(1, req.forecast_days+1):
        d = last_date + pd.Timedelta(days=i)
        for cat in cats:
            sub = history[history['category']==cat]
            # базовые признаки
            dow = d.dayofweek
            day = d.day
            row = {
                'category': cat,
                'day': day,
                'month': d.month,
                'dow': dow,
                'is_weekend': int(dow in [5,6])
            }
            # фурье-признаки
            import numpy as np
            row['sin_dow'] = np.sin(2 * np.pi * dow / 7)
            row['cos_dow'] = np.cos(2 * np.pi * dow / 7)
            row['sin_dom'] = np.sin(2 * np.pi * (day - 1) / 30)
            row['cos_dom'] = np.cos(2 * np.pi * (day - 1) / 30)
            # лаги
            for lag in [1,2,3,7,14,30]:
                row[f'lag_{lag}'] = sub['amount'].iloc[-lag] if len(sub)>=lag else 0
            # скользящие суммы/средние
            roll7 = sub['amount'].shift(1).rolling(7, min_periods=1)
            row['roll_7_sum'] = roll7.sum().iloc[-1]
            row['roll_7_mean'] = roll7.mean().iloc[-1]
            roll30 = sub['amount'].shift(1).rolling(30, min_periods=1)
            row['roll_30_sum'] = roll30.sum().iloc[-1]
            row['roll_30_mean'] = roll30.mean().iloc[-1]
            # количество дней с тратами
            row['roll_7_count'] = roll7.apply(lambda y: (y>0).sum(), raw=True).iloc[-1]
            row['roll_30_count'] = roll30.apply(lambda y: (y>0).sum(), raw=True).iloc[-1]
            # перцентиль
            row['roll_30_p75'] = roll30.quantile(0.75).iloc[-1]
            # days_since
            last_nz = sub[sub['amount']>0]['date'].max() if not sub[sub['amount']>0].empty else d
            row['days_since'] = int((d - last_nz).days)
            # предсказание
            Xf = pd.DataFrame([row])[feats]
            Xf['category'] = Xf['category'].astype('category')
            p = clf.predict_proba(Xf)[0,1]
            r = reg.predict(Xf)[0]
            val = max(0.0, float(p * r))
            # обновляем историю
            new_entry = {**row, 'amount': val, 'date': d}
            history = pd.concat([history, pd.DataFrame([new_entry])], ignore_index=True)
            # добавляем в результат
            future.append({'date': d.strftime('%Y-%m-%d'), 'category': cat, 'value': round(val,2)})
    
    return ForecastResponse(predictions=future)(predictions=future)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=80)
