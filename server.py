# server.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
from math import sqrt

app = FastAPI(title="Expense Forecast Service")

@app.post("/forecast")
async def forecast(file: UploadFile = File(...)):
    # 1. Прочитать CSV из запроса
    try:
        df = pd.read_csv(file.file, parse_dates=['date'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Can't read CSV: {e}")

    # 2. Ресемплинг по дням + нули
    all_cats = df['category'].unique()
    frames = []
    for cat in all_cats:
        tmp = (
            df[df['category'] == cat]
            .set_index('date')
            .resample('D')
            .agg({'amount': 'sum'})
        )
        tmp['category'] = cat
        tmp['amount'] = tmp['amount'].fillna(0)
        tmp = tmp.reset_index()
        frames.append(tmp)
    df_full = pd.concat(frames, ignore_index=True)

    # 3. Признаки даты
    df_full['day']        = df_full['date'].dt.day
    df_full['month']      = df_full['date'].dt.month
    df_full['dayofweek']  = df_full['date'].dt.dayofweek
    df_full['is_weekend']= df_full['dayofweek'].isin([5,6]).astype(int)

    # 4. Лаги и rolling-статистики
    df_full = df_full.sort_values(['category','date'])
    df_full['lag_1']   = df_full.groupby('category')['amount'].shift(1).fillna(0)
    df_full['lag_7']   = df_full.groupby('category')['amount'].shift(7).fillna(0)
    df_full['rolling_30_mean'] = (
        df_full.groupby('category')['amount']
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    )
    df_full['rolling_30_std'] = (
        df_full.groupby('category')['amount']
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).std().fillna(0))
    )
    # дни с последней ненулевой тратой
    df_full['last_nonzero_date'] = (
        df_full.where(df_full['amount']>0, pd.NaT)['date']
        .groupby(df_full['category']).ffill()
    )
    df_full['days_since_last'] = (
        (df_full['date'] - df_full['last_nonzero_date'])
        .dt.days.fillna(999).astype(int)
    )

    # 5. Бинарная цель
    df_full['target_bin'] = (df_full['amount']>0).astype(int)

    # 6. Фичи и категории
    features = [
        'category','day','month','dayofweek','is_weekend',
        'lag_1','lag_7','rolling_30_mean','rolling_30_std',
        'days_since_last'
    ]
    cat_features = ['category']

    X = df_full[features]
    y_bin = df_full['target_bin']
    mask = y_bin == 1
    y_amt = df_full.loc[mask, 'amount']

    # 7a. Обучаем классификатор
    clf = CatBoostClassifier(
        iterations=200, learning_rate=0.1, depth=6,
        random_seed=42, verbose=False
    )
    clf.fit(X, y_bin, cat_features=cat_features)

    # 7b. Обучаем регрессор только по ненулевым дням
    reg = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6,
        l2_leaf_reg=5, random_seed=42, verbose=False
    )
    reg.fit(X.loc[mask], y_amt, cat_features=cat_features)

    # 8. Делаем прогнозы по всем строкам
    pred_bin = clf.predict(X)
    pred_amt = reg.predict(X)
    pred_amt = np.where(pred_bin==1, pred_amt, 0.0)

    df_full['pred_amount'] = pred_amt

    # 9. Считаем метрики
    acc  = accuracy_score(y_bin, pred_bin)
    auc  = roc_auc_score(y_bin, clf.predict_proba(X)[:,1])
    mse = mean_squared_error(df_full['amount'], pred_amt)
    rmse = sqrt(mse)
    mae  = mean_absolute_error(df_full['amount'], pred_amt)

    # 10. Формируем вывод
    preds = df_full[['date','category','pred_amount']].copy()
    # сериализуем дату в строку
    preds['date'] = preds['date'].dt.strftime("%Y-%m-%d")
    records = preds.to_dict(orient='records')

    return JSONResponse({
        "metrics": {
            "acc":  round(acc,3),
            "auc":  round(auc,3),
            "rmse": round(rmse,2),
            "mae":  round(mae,2)
        },
        "predictions": records
    })


# точка старта, если нужно запускать как скрипт
if __name__=="__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
