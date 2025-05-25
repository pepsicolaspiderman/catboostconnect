from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

app = FastAPI(title="Сервис прогнозирования трат по категориям")

# Pydantic-модели
def build_transaction_models():
    class Transaction(BaseModel):
        date: str  # "YYYY-MM-DD"
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
    return Transaction, ForecastRequest, Prediction, ForecastResponse

Transaction, ForecastRequest, Prediction, ForecastResponse = build_transaction_models()

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # 1. Собираем историю
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="Список транзакций пуст")
    df['date'] = pd.to_datetime(df['date'])
    cats = df['category'].unique()

    # 2. Ресемплинг и фичи на всю историю + будущие дни
    all_frames = []
    for cat in cats:
        sub = df[df['category']==cat].set_index('date')['amount']
        series = sub.resample('D').sum().rename('amount')
        # добавляем будущие дни с NaN
        last = series.index.max()
        future_idx = pd.date_range(last + pd.Timedelta(days=1), periods=req.forecast_days, freq='D')
        series = pd.concat([series, pd.Series(data=[0]*len(future_idx), index=future_idx, name='amount')])
        frame = series.to_frame()
        frame['category'] = cat
        frame['day'] = frame.index.day
        frame['month'] = frame.index.month
        frame['dow'] = frame.index.dayofweek
        frame['is_weekend'] = frame['dow'].isin([5,6]).astype(int)
        # лаги и скользящие средние
        frame['lag_1'] = frame['amount'].shift(1).fillna(0)
        frame['lag_7'] = frame['amount'].shift(7).fillna(0)
        frame['rolling_30_mean'] = frame['amount'].shift(1).rolling(30, min_periods=1).mean()
        frame['rolling_30_std'] = frame['amount'].shift(1).rolling(30, min_periods=1).std().fillna(0)
        frame = frame.reset_index().rename(columns={'index':'date'})
        all_frames.append(frame)
    full = pd.concat(all_frames, ignore_index=True)

    # 3. Метки для обучения только на исторических датах
    full['is_future'] = full['date'] > full['date'].max() - pd.Timedelta(days=req.forecast_days)
    train = full[~full['is_future']]
    X_train = train[['category','day','month','dow','is_weekend','lag_1','lag_7','rolling_30_mean','rolling_30_std']].copy()
    X_train['category'] = X_train['category'].astype('category')
    y_bin = (train['amount']>0).astype(int)
    y_amt = train.loc[y_bin==1,'amount']

    # 4. Обучаем CatBoostClassifier и CatBoostRegressor как в вашем скрипте
    clf = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6,
                             random_seed=42, verbose=False, allow_writing_files=False)
    clf.fit(X_train, y_bin, cat_features=['category'])
    reg = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                            l2_leaf_reg=5, random_seed=42,
                            verbose=False, allow_writing_files=False)
    reg.fit(X_train[y_bin==1], y_amt, cat_features=['category'])

    # 5. Предсказание на всех датах (история+будущее)
    X_pred = full[['category','day','month','dow','is_weekend','lag_1','lag_7','rolling_30_mean','rolling_30_std']].copy()
    X_pred['category'] = X_pred['category'].astype('category')
    p = clf.predict_proba(X_pred)[:,1]
    r = reg.predict(X_pred)
    full['pred'] = (p * r).round(2)

    # 6. Собираем только будущие прогнозы
    future = full[full['is_future']]
    out = []
    for _, row in future.iterrows():
        out.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'category': row['category'],
            'value': row['pred']
        })

    return ForecastResponse(predictions=out)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=80)
