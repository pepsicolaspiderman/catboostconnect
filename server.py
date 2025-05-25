from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
from math import sqrt

app = FastAPI(title="Expense Forecast Service")

# Pydantic models
class Transaction(BaseModel):
    date: str        # "YYYY-MM-DD"
    category: str
    amount: float

class ForecastRequest(BaseModel):
    transactions: List[Transaction]
    forecast_days: int = Field(..., gt=0, description="Число дней вперёд для прогноза")

class Prediction(BaseModel):
    date: str
    value: float

class ForecastResponse(BaseModel):
    metrics: Dict[str, Dict[str, float]]
    predictions: List[Prediction]

# Global exception handler to return error details
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # 1. Build DataFrame
    df = pd.DataFrame([t.dict() for t in req.transactions])
    if df.empty:
        raise HTTPException(status_code=400, detail="transactions list is empty")
    df['date'] = pd.to_datetime(df['date'])

    # 2. Resample per day and fill zeros
    all_cats = df['category'].unique()
    frames = []
    for cat in all_cats:
        tmp = (
            df[df['category'] == cat]
              .set_index('date')['amount']
              .resample('D').sum()
              .to_frame()
        )
        tmp['category'] = cat
        tmp['amount'] = tmp['amount'].fillna(0)
        tmp = tmp.reset_index()
        frames.append(tmp)
    df_full = pd.concat(frames, ignore_index=True)

    # 3. Feature engineering
    df_full['category'] = df_full['category'].astype('category')
    df_full['day']       = df_full['date'].dt.day
    df_full['month']     = df_full['date'].dt.month
    df_full['dayofweek'] = df_full['date'].dt.dayofweek
    df_full['is_weekend']= df_full['dayofweek'].isin([5,6]).astype(int)

    df_full = df_full.sort_values(['category','date'])
    df_full['lag_1'] = df_full.groupby('category')['amount'].shift(1).fillna(0)
    df_full['lag_7'] = df_full.groupby('category')['amount'].shift(7).fillna(0)
    df_full['rolling_30_mean'] = (
        df_full.groupby('category')['amount']
               .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    )
    df_full['rolling_30_std'] = (
        df_full.groupby('category')['amount']
               .transform(lambda x: x.shift(1).rolling(30, min_periods=1).std().fillna(0))
    )
    df_full['last_nonzero_date'] = df_full['date'].where(df_full['amount'] > 0)
    df_full['last_nonzero_date'] = df_full.groupby('category')['last_nonzero_date'].ffill()
    df_full['days_since_last'] = (
        (df_full['date'] - df_full['last_nonzero_date']).dt.days.fillna(999).astype(int)
    )

    # 4. Prepare training data
    df_full['target_bin'] = (df_full['amount'] > 0).astype(int)
    features = [
        'category','day','month','dayofweek','is_weekend',
        'lag_1','lag_7','rolling_30_mean','rolling_30_std','days_since_last'
    ]
    cat_features = ['category']

    X = df_full[features]
    y_bin = df_full['target_bin']
    mask = y_bin == 1
    y_amt = df_full.loc[mask, 'amount']

    # 5. Train models (CatBoost)
    clf = CatBoostClassifier(
        iterations=200, learning_rate=0.1,
        depth=6, random_seed=42, verbose=False,
        allow_writing_files=False
    )
    clf.fit(X, y_bin, cat_features=cat_features)

    reg = CatBoostRegressor(
        iterations=300, learning_rate=0.05,
        depth=6, l2_leaf_reg=5,
        random_seed=42, verbose=False,
        allow_writing_files=False
    )
    reg.fit(X.loc[mask], y_amt, cat_features=cat_features)

    # 6. In-sample metrics
    pred_bin = clf.predict(X)
    pred_amt = reg.predict(X)
    pred_amt = np.where(pred_bin == 1, pred_amt, 0.0)

    acc  = accuracy_score(y_bin, pred_bin)
    auc  = roc_auc_score(y_bin, clf.predict_proba(X)[:, 1])
    mse  = mean_squared_error(df_full['amount'], pred_amt)
    rmse = sqrt(mse)
    mae  = mean_absolute_error(df_full['amount'], pred_amt)

    metrics = {
        'in-sample': {
            'Acc': round(acc, 3),
            'AUC': round(auc, 3),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2)
        }
    }

    # 7. Forecast future days
    last_date = df_full['date'].max()
    temp_df = df_full.copy()
    future = []
    for i in range(1, req.forecast_days + 1):
        d = last_date + pd.Timedelta(days=i)
        for cat in all_cats:
            subset = temp_df[temp_df['category'] == cat]
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
                'days_since_last': int((d - (subset[subset['amount'] > 0]['date'].max() if not subset[subset['amount'] > 0].empty else d)).days)
            }
            X_new = pd.DataFrame([feat])[features]
            X_new['category'] = X_new['category'].astype('category')
            bin_p = clf.predict(X_new)[0]
            amt_p = float(reg.predict(X_new)[0]) if bin_p == 1 else 0.0

            future.append({
                'date': d.strftime("%Y-%m-%d"),
                'value': round(amt_p, 2)
            })

            new_row = {**feat, 'amount': amt_p, 'target_bin': bin_p, 'date': d}
            new_rec = pd.DataFrame([new_row])
            new_rec['date'] = pd.to_datetime(new_rec['date'])
            temp_df = pd.concat([temp_df, new_rec], ignore_index=True)

    return ForecastResponse(metrics=metrics, predictions=future)

# Run on port 80
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=80)
