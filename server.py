from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
from catboost import CatBoostRegressor

app = FastAPI(title="Сервис прогнозирования трат по категориям")

def get_freq_threshold() -> float:
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

    # 2. Ресемплинг по дням
    categories = df['category'].unique()
    hist_frames = []
    for cat in categories:
        tmp = (
            df[df['category'] == cat]
              .set_index('date')['amount']
              .resample('D').sum()
              .to_frame()
        )
        tmp['category'] = cat
        tmp['amount'] = tmp['amount'].fillna(0)
        hist_frames.append(tmp.reset_index())
    hist = pd.concat(hist_frames, ignore_index=True)
    hist = hist.sort_values(['category', 'date']).reset_index(drop=True)

    # 3. Статистика по категориям
    freq = hist.groupby('category').apply(lambda x: (x['amount'] > 0).mean()).to_dict()
    # вычисляем моду дня платежа для редких
    modes = hist[hist['amount'] > 0].groupby('category')['date'] \
                .agg(lambda x: x.dt.day.mode().iat[0] if not x.dt.day.mode().empty else 1)\
                .to_dict()

    # 4. Генерация признаков
    hist['day'] = hist['date'].dt.day
    hist['month'] = hist['date'].dt.month
    hist['dow'] = hist['date'].dt.dayofweek
    hist['is_weekend'] = hist['dow'].isin([5,6]).astype(int)
    for lag in [1,7,30]:
        hist[f'lag_{lag}'] = hist.groupby('category')['amount'].shift(lag).fillna(0)
    hist['roll30_mean'] = hist.groupby('category')['amount'] \
                         .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    hist['roll30_std'] = hist.groupby('category')['amount'] \
                        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).std().fillna(0))
    hist['last_nz'] = hist['date'].where(hist['amount'] > 0)
    hist['last_nz'] = hist.groupby('category')['last_nz'].ffill()
    hist['days_since'] = (hist['date'] - hist['last_nz']).dt.days.fillna(999).astype(int)

    features = ['category','day','month','dow','is_weekend',
                'lag_1','lag_7','lag_30','roll30_mean','roll30_std','days_since']

    # 5. Обучение регрессора на всех данных
    X = hist[features].copy()
    X['category'] = X['category'].astype('category')
    y = hist['amount']
    reg = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    reg.fit(X, y, cat_features=['category'])

    # 6. Прогнозирование
    last_date = hist['date'].max()
    future = []
    freq_threshold = get_freq_threshold()

    # Будем эмулировать добавление прогнозов для поддержки лагов
    hist_future = hist.copy()

    for day_offset in range(1, req.forecast_days + 1):
        current_date = last_date + pd.Timedelta(days=day_offset)
        for cat in categories:
            cat_freq = freq.get(cat, 0)
            if cat_freq < freq_threshold:
                # редкие: только в свой типичный день
                if current_date.day == modes.get(cat, 1):
                    val = float(hist_future[(hist_future['category']==cat)]['amount'].tail(30).sum())
                else:
                    val = 0.0
            else:
                # частые: используем регрессию
                sub = hist_future[hist_future['category']==cat]
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
                    'days_since': int((current_date - sub[sub['amount']>0]['date'].max()).days)
                                  if not sub[sub['amount']>0].empty else 999
                }
                Xf = pd.DataFrame([row])[features]
                Xf['category'] = Xf['category'].astype('category')
                pred = reg.predict(Xf)[0]
                val = max(0.0, float(pred))
            # добавляем в историю
            new_record = row.copy() if cat_freq >= freq_threshold else {'category':cat,'day':current_date.day,'month':current_date.month,'dow':current_date.dayofweek,'is_weekend':int(current_date.dayofweek in [5,6]),'lag_1':0,'lag_7':0,'lag_30':0,'roll30_mean':0,'roll30_std':0,'days_since':0}
            new_record.update({'date':current_date,'amount':val})
            hist_future = pd.concat([hist_future, pd.DataFrame([new_record])], ignore_index=True)
            future.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'category': cat,
                'value': round(val,2)
            })

    return ForecastResponse(predictions=future)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=80)
