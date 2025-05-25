from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

app = FastAPI(title="Сервис прогнозирования трат по категориям")

# Pydantic models
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

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # 1. Load and prepare input data
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="Список транзакций пуст")
    df['date'] = pd.to_datetime(df['date'])
    cats = df['category'].unique()

    # 2. Build full daily series including future days placeholder
    frames = []
    for cat in cats:
        series = df[df['category']==cat].set_index('date')['amount'].resample('D').sum()
        last = series.index.max()
        future_idx = pd.date_range(last + pd.Timedelta(days=1), periods=req.forecast_days, freq='D')
        series = pd.concat([series, pd.Series(0, index=future_idx)])
        frame = series.to_frame(name='amount')
        frame['category'] = cat
        frame = frame.reset_index().rename(columns={'index':'date'})
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True).sort_values(['category','date']).reset_index(drop=True)

    # 3. Label future vs train
    full['is_future'] = full.groupby('category')['date'].transform(lambda x: x > x.max() - pd.Timedelta(days=req.forecast_days))
    train = full[~full['is_future']].copy()

    # 4. Feature generation
    def gen_feats(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['dow'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
        df['lag_1'] = df.groupby('category')['amount'].shift(1).fillna(0)
        df['lag_7'] = df.groupby('category')['amount'].shift(7).fillna(0)
        df['rolling_30_mean'] = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        df['rolling_30_std']  = df.groupby('category')['amount'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).std().fillna(0))
        return df

    full = gen_feats(full)
    train = full[~full['is_future']]
    X_train = train[['category','day','month','dow','is_weekend','lag_1','lag_7','rolling_30_mean','rolling_30_std']].copy()
    X_train['category'] = X_train['category'].astype('category')
    y_bin = (train['amount']>0).astype(int)
    y_amt = train.loc[y_bin==1,'amount']

    # 5. Train classifier and regressor
    clf = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6,
                             random_seed=42, verbose=False, allow_writing_files=False)
    clf.fit(X_train, y_bin, cat_features=['category'])
    reg = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                            l2_leaf_reg=5, random_seed=42,
                            verbose=False, allow_writing_files=False)
    reg.fit(X_train[y_bin==1], y_amt, cat_features=['category'])

    # 6. Rare vs frequent categorization
    freq = train.groupby('category').apply(lambda x: (x['amount']>0).mean())
    modes = (train[train['amount']>0].groupby('category')['date']
             .agg(lambda x: x.dt.day.mode().iat[0] if not x.dt.day.mode().empty else 1))
    sum30 = train.groupby('category')['amount'].apply(lambda x: x.tail(30).sum())

    # 7. Predict future
    feats = ['category','day','month','dow','is_weekend','lag_1','lag_7','rolling_30_mean','rolling_30_std']
    out = []
    for cat in cats:
        sub = full[full['category']==cat]
        cat_freq = freq.get(cat, 0)
        cat_mode = modes.get(cat, 1)
        cat_sum30 = sum30.get(cat, 0)
        for _, row in sub[sub['is_future']].iterrows():
            d = row['date']
            if cat_freq < 0.1:
                # rare: only pay once on typical day
                val = round(cat_sum30, 2) if d.day==cat_mode else 0.0
            else:
                # frequent: use p * r
                Xf = row[feats].to_frame().T
                Xf['category'] = Xf['category'].astype('category')
                p = clf.predict_proba(Xf)[0,1]
                r = reg.predict(Xf)[0]
                val = max(0.0, float(round(p*r,2)))
            out.append({'date': d.strftime('%Y-%m-%d'), 'category': cat, 'value': val})

    return ForecastResponse(predictions=out)

# Pydantic-модели для рекомендаций
class RecommendationsRequest(BaseModel):
    avg_totals: Dict[str, float]
    last_totals: Dict[str, float]

class RecommendationItem(BaseModel):
    category: str
    current: float
    benchmark: float
    advice: str

class RecommendationsResponse(BaseModel):
    recommendations: List[RecommendationItem]

# Новый endpoint для рекомендаций
@app.post("/recommendations", response_model=RecommendationsResponse)
def recommendations(req: RecommendationsRequest):
    recommendations: List[RecommendationItem] = []
    # Порог, выше которого даём совет сократить расходы
    threshold = 1.1  # 10% выше среднего
    for cat, current in req.last_totals.items():
        benchmark = req.avg_totals.get(cat, 0.0)
        if benchmark > 0 and current > benchmark * threshold:
            reduction = (current - benchmark) / current
            percent = round(reduction * 100)
            advice = (
                f"Постарайтесь сократить траты на «{cat}» примерно на {percent}% "
            )
            recommendations.append(RecommendationItem(
                category=cat,
                current=current,
                benchmark=benchmark,
                advice=advice
            ))
    return RecommendationsResponse(recommendations=recommendations)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=80)