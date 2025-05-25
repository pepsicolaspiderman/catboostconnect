from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
from catboost import CatBoostRegressor

app = FastAPI(title="Сервис прогнозирования трат по категориям")

# Pydantic-модели
default_freq_threshold = 0.3  # порог доли активных дней для разделения частых и редких

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
    # 1. Подготовка данных
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="Список транзакций пуст")
    df['date'] = pd.to_datetime(df['date'])

    # 2. Ресемплинг по дням на всю историю
    categories = df['category'].unique()
    frames = []
    for cat in categories:
        tmp = df[df['category']==cat].set_index('date')['amount'].resample('D').sum().to_frame()
        tmp['category'] = cat
        tmp['amount'] = tmp['amount'].fillna(0)
        frames.append(tmp.reset_index())
    hist = pd.concat(frames, ignore_index=True).sort_values(['category','date'])

    # 3. Статистика по категории: частота ненулевых дней
    freq = hist.groupby('category').apply(lambda x: (x['amount']>0).mean()).to_dict()

    # 4. Создание фичей для регрессии на все дни
    hist['day'] = hist['date'].dt.day
    hist['month'] = hist['date'].dt.month
    hist['dow'] = hist['date'].dt.dayofweek
    hist['is_weekend'] = hist['dow'].isin([5,6]).astype(int)
    hist = hist.sort_values(['category','date'])
    for lag in [1,7,30]:
        hist[f'lag_{lag}'] = hist.groupby('category')['amount'].shift(lag).fillna(0)
    hist['roll30_mean'] = hist.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30,min_periods=1).mean())
    hist['roll30_std'] = hist.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30,min_periods=1).std().fillna(0))
    hist['last_nz'] = hist['date'].where(hist['amount']>0)
    hist['last_nz'] = hist.groupby('category')['last_nz'].ffill()
    hist['days_since'] = (hist['date']-hist['last_nz']).dt.days.fillna(999).astype(int)

    features = ['category','day','month','dow','is_weekend','lag_1','lag_7','lag_30','roll30_mean','roll30_std','days_since']
    hist_X = hist[features].copy()
    hist_X['category'] = hist_X['category'].astype('category')
    hist_y = hist['amount']

    # 5. Обучение одного регрессора на все данные
    reg = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                            l2_leaf_reg=5, random_seed=42,
                            verbose=False, allow_writing_files=False)
    reg.fit(hist_X, hist_y, cat_features=['category'])

    # 6. Вычисление типичного дня для редких платежей
    modes = hist[hist['amount']>0].groupby('category')['day'].agg(lambda x: x.mode().iat[0] if not x.mode().empty else 1).to_dict()

    # 7. Прогнозирование
    last_date = hist['date'].max()
    future = []
    for i in range(1, req.forecast_days+1):
        d = last_date + pd.Timedelta(days=i)
        for cat in categories:
            cat_freq = freq.get(cat,0)
            if cat_freq < default_freq_threshold:
                # редкий: прогноз только в типичный день суммой за прошлые N дней
                md = modes.get(cat,1)
                if d.day==md:
                    past_sum = hist[(hist['category']==cat)&(hist['date']<=last_date)]['amount'].tail(30).sum()
                    val = round(past_sum,2)
                else:
                    val = 0.0
            else:
                # частый: ежедневный регрессор
                row = {
                    'category': cat,
                    'day': d.day,'month': d.month,'dow': d.dayofweek,
                    'is_weekend': int(d.dayofweek in [5,6]),
                    'lag_1': hist[(hist['category']==cat) & (hist['date']==d-pd.Timedelta(days=1))]['amount'].iat[0] if not hist[(hist['category']==cat) & (hist['date']==d-pd.Timedelta(days=1))].empty else 0,
                    'lag_7': hist[(hist['category']==cat) & (hist['date']==d-pd.Timedelta(days=7))]['amount'].iat[0] if not hist[(hist['category']==cat) & (hist['date']==d-pd.Timedelta(days=7))].empty else 0,
                    'lag_30': hist[(hist['category']==cat) & (hist['date']==d-pd.Timedelta(days=30))]['amount'].iat[0] if not hist[(hist['category']==cat) & (hist['date']==d-pd.Timedelta(days=30))].empty else 0,
                    'roll30_mean': hist[hist['category']==cat]['amount'].shift(1).rolling(30,min_periods=1).mean().iloc[-1],
                    'roll30_std': hist[hist['category']==cat]['amount'].shift(1).rolling(30,min_periods=1).std().fillna(0).iloc[-1],
                    'days_since': int((d - hist[(hist['category']==cat)&(hist['amount']>0)]['date'].max()).days) if not hist[(hist['category']==cat)&(hist['amount']>0)].empty else 999
                }
                Xf = pd.DataFrame([row])[features]
                Xf['category'] = Xf['category'].astype('category')
                val = float(reg.predict(Xf)[0])
            future.append({'date':d.strftime('%Y-%m-%d'),'category':cat,'value':val})
    return ForecastResponse(predictions=future)

if __name__=='__main__':
    import uvicorn
    uvicorn.run('server:app',host='0.0.0.0',port=80)
