from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
from math import sqrt

app = FastAPI(title="Сервис прогнозирования трат")

# Pydantic-модели
class Transaction(BaseModel):
    date: str        # "YYYY-MM-DD"
    category: str
    amount: float

class ForecastRequest(BaseModel):
    transactions: List[Transaction]
    forecast_days: int = Field(..., gt=0, description="Число дней для прогноза")

class Prediction(BaseModel):
    date: str       # Дата прогноза
    value: float    # Прогнозируемая сумма траты

class ForecastResponse(BaseModel):
    metrics: Dict[str, Dict[str, float]]   # Метрики качества модели
    predictions: List[Prediction]          # Список прогнозов

# Глобальный обработчик исключений
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # 1. Построение DataFrame из входного JSON
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="Список транзакций пуст")
    df['date'] = pd.to_datetime(df['date'])

    # 2. Ресемплинг по дням и заполнение нулей
    categories = df['category'].unique()
    frames = []
    for cat in categories:
        tmp = (
            df[df['category']==cat]
              .set_index('date')['amount']
              .resample('D').sum()
              .to_frame()
        )
        tmp['category'] = cat
        tmp['amount'] = tmp['amount'].fillna(0)
        frames.append(tmp.reset_index())
    df_full = pd.concat(frames, ignore_index=True)

    # 3. Признаки
    df_full['category'] = df_full['category'].astype('category')
    df_full['day'] = df_full['date'].dt.day
    df_full['month'] = df_full['date'].dt.month
    df_full['dow'] = df_full['date'].dt.dayofweek
    df_full['is_weekend'] = df_full['dow'].isin([5,6]).astype(int)
    df_full = df_full.sort_values(['category','date'])
    df_full['lag_1'] = df_full.groupby('category')['amount'].shift(1).fillna(0)
    df_full['lag_7'] = df_full.groupby('category')['amount'].shift(7).fillna(0)
    df_full['roll30_mean'] = (
        df_full.groupby('category')['amount']
               .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    )
    df_full['roll30_std'] = (
        df_full.groupby('category')['amount']
               .transform(lambda x: x.shift(1).rolling(30, min_periods=1).std().fillna(0))
    )
    df_full['last_nz'] = df_full['date'].where(df_full['amount']>0)
    df_full['last_nz'] = df_full.groupby('category')['last_nz'].ffill()
    df_full['days_since'] = (df_full['date'] - df_full['last_nz']).dt.days.fillna(999).astype(int)

    # 4. Подготовка данных для классификации и регрессии
    df_full['target_bin'] = (df_full['amount']>0).astype(int)
    features = ['category','day','month','dow','is_weekend','lag_1','lag_7','roll30_mean','roll30_std','days_since']
    cat_feats = ['category']
    X = df_full[features]
    y_bin = df_full['target_bin']
    mask = y_bin==1
    y_amt = df_full.loc[mask,'amount']

    # 5. Обучение моделей
    clf = CatBoostClassifier(
        iterations=200, learning_rate=0.1, depth=6,
        random_seed=42, verbose=False, allow_writing_files=False
    )
    clf.fit(X, y_bin, cat_features=cat_feats)

    reg = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=5,
        random_seed=42, verbose=False, allow_writing_files=False
    )
    reg.fit(X.loc[mask], y_amt, cat_features=cat_feats)

    # 6. Метрики in-sample
    pred_bin = clf.predict(X)
    pred_prob = clf.predict_proba(X)[:,1]
    pred_reg = reg.predict(X)
    combined = pred_reg * pred_prob

    acc = accuracy_score(y_bin, pred_bin)
    auc = roc_auc_score(y_bin, pred_prob)
    mse = mean_squared_error(df_full['amount'], combined)
    rmse = sqrt(mse)
    mae = mean_absolute_error(df_full['amount'], combined)
    metrics = {'in-sample': {'Acc':round(acc,3),'AUC':round(auc,3),'RMSE':round(rmse,2),'MAE':round(mae,2)}}

    # 7. Прогноз
    last_date = df_full['date'].max()
    hist = df_full.copy()
    future = []
    for i in range(1, req.forecast_days+1):
        d = last_date + pd.Timedelta(days=i)
        for cat in categories:
            sub = hist[hist['category']==cat]
            row = {
                'category':cat,
                'day':d.day,'month':d.month,'dow':d.dayofweek,
                'is_weekend':int(d.dayofweek in [5,6]),
                'lag_1':sub['amount'].iloc[-1],
                'lag_7':sub['amount'].shift(6).iloc[-1] if len(sub)>=7 else 0,
                'roll30_mean':sub['amount'].shift(1).rolling(30, min_periods=1).mean().iloc[-1],
                'roll30_std':sub['amount'].shift(1).rolling(30, min_periods=1).std().fillna(0).iloc[-1],
                'days_since':int((d - (sub[sub['amount']>0]['date'].max() if not sub[sub['amount']>0].empty else d)).days)
            }
            Xf = pd.DataFrame([row])[features]
            Xf['category'] = Xf['category'].astype('category')
            p = clf.predict_proba(Xf)[0,1]
            r = float(reg.predict(Xf)[0])
            val = p * r
            future.append({'date':d.strftime('%Y-%m-%d'),'value':round(val,2)})
            new = {**row,'amount':val,'date':d}
            hist = pd.concat([hist,pd.DataFrame([new])],ignore_index=True)

    return ForecastResponse(metrics=metrics,predictions=future)

if __name__=='__main__':
    import uvicorn
    uvicorn.run('server:app',host='0.0.0.0',port=80)
