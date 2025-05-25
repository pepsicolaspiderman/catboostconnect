from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

app = FastAPI(title="Сервис прогнозирования трат")

# Модели для валидации входных данных и формирования ответа
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

# Глобальный обработчик исключений, возвращает текст ошибки
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # 1. Преобразуем входящие транзакции в DataFrame
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="Список транзакций пуст")
    df['date'] = pd.to_datetime(df['date'])

    # 2. Ресемплинг по дням с заполнением нулями
    categories = df['category'].unique()
    frames = []
    for cat in categories:
        tmp = df[df['category'] == cat]
        tmp = tmp.set_index('date')['amount'].resample('D').sum().to_frame()
        tmp['category'] = cat
        tmp['amount'] = tmp['amount'].fillna(0)
        frames.append(tmp.reset_index())
    df_full = pd.concat(frames, ignore_index=True)

    # 3. Создаем признаки
    df_full['category'] = df_full['category'].astype('category')
    df_full['day'] = df_full['date'].dt.day
    df_full['month'] = df_full['date'].dt.month
    df_full['dayofweek'] = df_full['date'].dt.dayofweek
    df_full['is_weekend'] = df_full['dayofweek'].isin([5,6]).astype(int)

    df_full = df_full.sort_values(['category','date'])
    df_full['lag_1'] = df_full.groupby('category')['amount'].shift(1).fillna(0)
    df_full['lag_7'] = df_full.groupby('category')['amount'].shift(7).fillna(0)
    df_full['rolling_30_mean'] = df_full.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    df_full['rolling_30_std'] = df_full.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).std().fillna(0))
    df_full['last_nonzero_date'] = df_full['date'].where(df_full['amount'] > 0)
    df_full['last_nonzero_date'] = df_full.groupby('category')['last_nonzero_date'].ffill()
    df_full['days_since_last'] = (df_full['date'] - df_full['last_nonzero_date']).dt.days.fillna(999).astype(int)

    # 4. Обучение регрессора на всех данных (включая нули)
    features = ['category','day','month','dayofweek','is_weekend','lag_1','lag_7','rolling_30_mean','rolling_30_std','days_since_last']
    X = df_full[features]
    y = df_full['amount']
    cat_features = ['category']

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    model.fit(X, y, cat_features=cat_features)

    # 5. Рассчитываем метрики качества на обучающей выборке
    preds_in = model.predict(X)
    mse = mean_squared_error(y, preds_in)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y, preds_in)
    metrics = {'in-sample': {'RMSE': round(rmse,2), 'MAE': round(mae,2)}}

    # 6. Прогнозируем будущие дни
    last_date = df_full['date'].max()
    temp = df_full.copy()
    future = []
    for i in range(1, req.forecast_days + 1):
        d = last_date + pd.Timedelta(days=i)
        for cat in categories:
            subset = temp[temp['category'] == cat]
            feat = {
                'category': cat,
                'day': d.day,
                'month': d.month,
                'dayofweek': d.dayofweek,
                'is_weekend': int(d.dayofweek in [5,6]),
                'lag_1': subset['amount'].iloc[-1],
                'lag_7': subset['amount'].shift(6).iloc[-1] if len(subset) >= 7 else 0,
                'rolling_30_mean': subset['amount'].shift(1).rolling(30, min_periods=1).mean().iloc[-1],
                'rolling_30_std': subset['amount'].shift(1).rolling(30, min_periods=1).std().fillna(0).iloc[-1],
                'days_since_last': int((d - (subset[subset['amount']>0]['date'].max() if not subset[subset['amount']>0].empty else d)).days)
            }
            # Формируем таблицу для предсказания и приводим категорию к типу 'category'
            X_new = pd.DataFrame([feat])[features]
            X_new['category'] = X_new['category'].astype('category')
            # Предсказанное значение траты
            val = float(model.predict(X_new)[0])
            future.append({'date': d.strftime('%Y-%m-%d'), 'value': round(val,2)})
            # Добавляем предсказание в temp для лагов следующих дней
            new_row = {**feat, 'amount': val, 'date': d}
            new_df = pd.DataFrame([new_row])
            new_df['category'] = new_df['category'].astype('category')
            new_df['date'] = pd.to_datetime(new_df['date'])
            temp = pd.concat([temp, new_df], ignore_index=True)

    return ForecastResponse(metrics=metrics, predictions=future)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=80)