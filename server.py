from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
from catboost import CatBoostRegressor

app = FastAPI(title="Сервис прогнозирования трат по категориям")

def get_freq_threshold():
    return 0.1  # порог доли активных дней для разделения частых и редких

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
    # 1. Загрузка и подготовка данных
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="Список транзакций пуст")
    df['date'] = pd.to_datetime(df['date'])

    # 2. Ресемплинг по дням на всю историю
    categories = df['category'].unique()
    frames = []
    for cat in categories:
        tmp = df[df['category'] == cat].set_index('date')['amount'].resample('D').sum().to_frame()
        tmp['category'] = cat
        frames.append(tmp.reset_index())
    hist = pd.concat(frames, ignore_index=True).fillna(0)
    hist = hist.sort_values(['category', 'date']).reset_index(drop=True)

    # 3. Статистика по категории: частота ненулевых дней и типичный день
    freq = hist.groupby('category').apply(lambda x: (x['amount'] > 0).mean()).to_dict()
    modes = hist[hist['amount'] > 0].groupby('category')['date'] \
                .agg(lambda x: x.dt.day.mode().iat[0] if not x.dt.day.mode().empty else 1).to_dict()

    # 4. Генерация общих признаков на всю историю
    hist['day'] = hist['date'].dt.day
    hist['month'] = hist['date'].dt.month
    hist['dow'] = hist['date'].dt.dayofweek
    hist['is_weekend'] = hist['dow'].isin([5,6]).astype(int)
    # лаги и статистики
    for lag in [1,7,30]:
        hist[f'lag_{lag}'] = hist.groupby('category')['amount'].shift(lag).fillna(0)
    hist['roll30_mean'] = hist.groupby('category')['amount'] \
                         .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    hist['roll30_std']  = hist.groupby('category')['amount'] \
                         .transform(lambda x: x.shift(1).rolling(30, min_periods=1).std().fillna(0))
    hist['last_nz'] = hist['date'].where(hist['amount'] > 0)
    hist['last_nz'] = hist.groupby('category')['last_nz'].ffill()
    hist['days_since'] = (hist['date'] - hist['last_nz']).dt.days.fillna(999).astype(int)

    features = ['category','day','month','dow','is_weekend',
                'lag_1','lag_7','lag_30','roll30_mean','roll30_std','days_since']

    # 5. Обучение регрессора на всех данных
    X_train = hist[features].copy()
    X_train['category'] = X_train['category'].astype('category')
    y_train = hist['amount']
    reg = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6,
        l2_leaf_reg=5, random_seed=42,
        verbose=False, allow_writing_files=False
    )
    reg.fit(X_train, y_train, cat_features=['category'])

    # 6. Прогнозирование с динамическим обновлением истории
    last_date = hist['date'].max()
    future = []
    freq_threshold = get_freq_threshold()

    for i in range(1, req.forecast_days + 1):
        current_date = last_date + pd.Timedelta(days=i)
        for cat in categories:
            cat_freq = freq.get(cat, 0)
            if cat_freq < freq_threshold:
                # Редкие: только в типичный день месячной суммы
                if current_date.day == modes.get(cat, 1):
                    past_sum = hist[(hist['category']==cat)]['amount'].tail(30).sum()
                    val = round(past_sum, 2)
                else:
                    val = 0.0
            else:
                # Частые: предсказание регрессором с актуальными фичами
                sub = hist[hist['category']==cat]
                row = {
                    'category': cat,
                    'day': current_date.day,
                    'month': current_date.month,
                    'dow': current_date.dayofweek,
                    'is_weekend': int(current_date.dayofweek in [5,6]),
                    'lag_1': sub['amount'].iloc[-1],
                    'lag_7': sub['amount'].shift(6).iloc[-1] if len(sub)>=7 else 0,
                    'lag_30': sub['amount'].shift(29).iloc[-1] if len(sub)>=30 else 0,
                    'roll30_mean': sub['amount'].shift(1).rolling(30, min_periods=1).mean().iloc[-1],
                    'roll30_std': sub['amount'].shift(1).rolling(30, min_periods=1).std().fillna(0).iloc[-1],
                    'days_since': int((current_date - sub[sub['amount']>0]['date'].max()).days) \
                                  if not sub[sub['amount']>0].empty else 999
                }
                Xf = pd.DataFrame([row])[features]
                Xf['category'] = Xf['category'].astype('category')
                val = float(reg.predict(Xf)[0])
                # Обрезаем отрицательные
                if val < 0:
                    val = 0.0
            # Добавляем прогноз в историю для следующих итераций
            new_entry = {
                'date': current_date,
                'category': cat,
                'amount': val,
                **{k: row[k] for k in row if k in row}
            }
            hist = pd.concat([hist, pd.DataFrame([new_entry])], ignore_index=True)
            future.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'category': cat,
                'value': val
            })
    return ForecastResponse(predictions=future)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=80)
