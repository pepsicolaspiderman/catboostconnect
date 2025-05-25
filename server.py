from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from catboost import CatBoostRegressor

app = FastAPI(title="Сервис прогнозирования трат по категориям")

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
    category: str   # Категория траты
    value: float    # Прогнозируемая сумма траты

class ForecastResponse(BaseModel):
    predictions: List[Prediction]

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # 1. Входные данные
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="Список транзакций пуст")
    df['date'] = pd.to_datetime(df['date'])

    # 2. Ресемплинг по дням с заполнением нулями
    categories = df['category'].unique()
    frames = []
    for cat in categories:
        tmp = (
            df[df['category'] == cat]
              .set_index('date')['amount']
              .resample('D').sum().to_frame()
        )
        tmp['category'] = cat
        tmp['amount'] = tmp['amount'].fillna(0)
        frames.append(tmp.reset_index())
    df_full = pd.concat(frames, ignore_index=True)

    # 3. Создание фичей с учётом месячной периодики
    df_full['day'] = df_full['date'].dt.day
    df_full['month'] = df_full['date'].dt.month
    df_full['dow'] = df_full['date'].dt.dayofweek
    df_full['is_weekend'] = df_full['dow'].isin([5,6]).astype(int)
    df_full = df_full.sort_values(['category','date'])
    # Лаги на 1, 7 и 30 дней
    df_full['lag_1'] = df_full.groupby('category')['amount'].shift(1).fillna(0)
    df_full['lag_7'] = df_full.groupby('category')['amount'].shift(7).fillna(0)
    df_full['lag_30'] = df_full.groupby('category')['amount'].shift(30).fillna(0)
    # Скользящие статистики
    df_full['roll30_mean'] = (
        df_full.groupby('category')['amount']
               .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    )
    df_full['roll30_std'] = (
        df_full.groupby('category')['amount']
               .transform(lambda x: x.shift(1).rolling(30, min_periods=1).std().fillna(0))
    )
    # Дней с последней ненулевой траты
    df_full['last_nz'] = df_full['date'].where(df_full['amount'] > 0)
    df_full['last_nz'] = df_full.groupby('category')['last_nz'].ffill()
    df_full['days_since'] = (df_full['date'] - df_full['last_nz']).dt.days.fillna(999).astype(int)

    # 4. Подготовка данных для регрессии на все дни (включая нули)
    features = [
        'category','day','month','dow','is_weekend',
        'lag_1','lag_7','lag_30','roll30_mean','roll30_std','days_since'
    ]
    cat_feats = ['category']
    X_full = df_full[features].copy()
    X_full['category'] = X_full['category'].astype('category')
    y_full = df_full['amount']

    # 5. Обучение одного регрессора на всех данных
    reg_all = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    reg_all.fit(X_full, y_full, cat_features=cat_feats)

    # 6. Прогноз на будущее
    last_date = df_full['date'].max()
    hist = df_full.copy()
    future_preds = []

    for i in range(1, req.forecast_days + 1):
        d = last_date + pd.Timedelta(days=i)
        for cat in categories:
            sub = hist[hist['category'] == cat]
            row = {
                'category': cat,
                'day': d.day,
                'month': d.month,
                'dow': d.dayofweek,
                'is_weekend': int(d.dayofweek in [5, 6]),
                'lag_1': sub['amount'].iloc[-1],
                'lag_7': sub['amount'].shift(6).iloc[-1] if len(sub) >= 7 else 0,
                'lag_30': sub['amount'].shift(29).iloc[-1] if len(sub) >= 30 else 0,
                'roll30_mean': sub['amount'].shift(1).rolling(30, min_periods=1).mean().iloc[-1],
                'roll30_std': sub['amount'].shift(1).rolling(30, min_periods=1).std().fillna(0).iloc[-1],
                'days_since': int((d - (sub[sub['amount']>0]['date'].max() if not sub[sub['amount']>0].empty else d)).days)
            }
            Xf = pd.DataFrame([row])[features]
            Xf['category'] = Xf['category'].astype('category')
            val = float(reg_all.predict(Xf)[0])

            future_preds.append({
                'date': d.strftime('%Y-%m-%d'),
                'category': cat,
                'value': round(val, 2)
            })

            new_entry = {**row, 'amount': val, 'date': d}
            hist = pd.concat([hist, pd.DataFrame([new_entry])], ignore_index=True)

    return ForecastResponse(predictions=future_preds)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=80)
