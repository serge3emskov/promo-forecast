from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import os

app = FastAPI(title="Promo Sales Forecast API")

# Определяем базовую директорию (для Docker это /app)
# Проверяем наличие файлов модели в разных возможных локациях
if os.path.exists('/app/promo_model.pkl'):
    BASE_DIR = '/app'
elif os.path.exists('promo_model.pkl'):
    BASE_DIR = '.'
else:
    # По умолчанию используем текущую директорию или /app если работаем в контейнере
    BASE_DIR = '/app' if os.path.exists('/app') else '.'

# Пути к файлам модели
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'promo_model.pkl'))
METADATA_PATH = os.getenv('METADATA_PATH', os.path.join(BASE_DIR, 'model_metadata.json'))
HISTORY_PATH = os.getenv('HISTORY_PATH', os.path.join(BASE_DIR, 'sales_history.csv'))

# Фоллбэк: если файлы не найдены по основному пути, ищем в корне /app
if not os.path.exists(MODEL_PATH) and os.path.exists('/app/promo_model.pkl'):
    MODEL_PATH = '/app/promo_model.pkl'
    METADATA_PATH = '/app/model_metadata.json'
    HISTORY_PATH = '/app/sales_history.csv'

# Загружаем модель
try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    print(f"Модель загружена из {MODEL_PATH}")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None
    metadata = None

feature_cols = metadata['feature_cols'] if metadata else []

class PromoItem(BaseModel):
    sku: str
    promo_start: str  # формат YYYY-MM-DD
    promo_end: str    # формат YYYY-MM-DD
    discount: float   # процент скидки
    promo_type: Optional[str] = "discount"

class ForecastRequest(BaseModel):
    items: List[PromoItem]
    forecast_days: int = 14

class ForecastItem(BaseModel):
    sku: str
    date: str
    predicted_sales: float
    is_promo: bool
    discount: float

class ForecastResponse(BaseModel):
    forecast: List[ForecastItem]
    model_mape: float
    generated_at: str
    total_forecast: float
    actual_stock: float
    order_quantity: float

def prepare_features(sku: str, date: pd.Timestamp, 
                     historical_data: pd.DataFrame,
                     promo_schedule: dict) -> pd.DataFrame:
    """Готовим признаки для прогноза"""
    
    # Базовые признаки
    features = {
        'is_promo': 0,
        'discount': 0,
        'day_of_week': date.dayofweek,
        'month': date.month,
        'is_weekend': 1 if date.dayofweek >= 5 else 0,
        'is_holiday': 0,  # можно расширить
        'lag_1': 0,
        'lag_7': 0,
        'lag_14': 0,
        'lag_28': 0,
        'rolling_mean_7': 0,
        'rolling_mean_14': 0,
        'rolling_std_7': 0,
        'days_since_promo': 0,
        'promo_freq_28': 0,
        'sku_encoded': 0  # нужно маппить SKU
    }
    
    # Проверяем промо
    if sku in promo_schedule:
        for promo in promo_schedule[sku]:
            if promo['start'] <= date <= promo['end']:
                features['is_promo'] = 1
                features['discount'] = promo['discount']
                break
    
    # Берем исторические данные для лагов
    sku_history = historical_data[historical_data['sku'] == sku].copy()
    if len(sku_history) > 0:
        days_before = (date - sku_history['date'].max()).days
        
        if len(sku_history) >= 1:
            features['lag_1'] = sku_history.iloc[-1]['sales']
        if len(sku_history) >= 7:
            features['lag_7'] = sku_history.iloc[-7]['sales']
        if len(sku_history) >= 14:
            features['lag_14'] = sku_history.iloc[-14]['sales']
        if len(sku_history) >= 28:
            features['lag_28'] = sku_history.iloc[-28]['sales']
            features['rolling_mean_7'] = sku_history.iloc[-28:-21]['sales'].mean()
            features['rolling_mean_14'] = sku_history.iloc[-28:-14]['sales'].mean()
            features['rolling_std_7'] = sku_history.iloc[-28:-21]['sales'].std()
    
    return pd.DataFrame([features])

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_promo_sales(request: ForecastRequest):
    """Прогноз продаж для промо-акций"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    # Загружаем исторические данные
    if not os.path.exists(HISTORY_PATH):
        raise HTTPException(status_code=400, detail="Исторические данные не найдены")
    
    historical_df = pd.read_csv(HISTORY_PATH)
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    
    # Создаем расписание промо
    promo_schedule = {}
    for item in request.items:
        if item.sku not in promo_schedule:
            promo_schedule[item.sku] = []
        promo_schedule[item.sku].append({
            'start': pd.to_datetime(item.promo_start),
            'end': pd.to_datetime(item.promo_end),
            'discount': item.discount
        })
    
    # Генерируем прогноз
    forecasts = []
    start_date = datetime.now().date()
    
    for item in request.items:
        for day_offset in range(request.forecast_days):
            forecast_date = pd.Timestamp(start_date) + timedelta(days=day_offset)
            
            # Готовим признаки
            features = prepare_features(
                sku=item.sku,
                date=forecast_date,
                historical_data=historical_df,
                promo_schedule=promo_schedule
            )
            
            # Предсказываем
            if features[feature_cols].isnull().any().any():
                features = features.fillna(0)
            
            prediction = model.predict(features[feature_cols])[0]
            prediction = max(0, prediction)  # продажи не могут быть отрицательными
            
            # Проверяем, промо ли это
            is_promo = False
            discount = 0
            if item.sku in promo_schedule:
                for promo in promo_schedule[item.sku]:
                    if promo['start'] <= forecast_date <= promo['end']:
                        is_promo = True
                        discount = promo['discount']
                        break
            
            forecasts.append(ForecastItem(
                sku=item.sku,
                date=forecast_date.strftime('%Y-%m-%d'),
                predicted_sales=round(prediction, 2),
                is_promo=is_promo,
                discount=discount
            ))
    
    # Считаем сумму прогноза за период промо
    total_forecast = sum(f.predicted_sales for f in forecasts if f.is_promo)
    
    # Получаем фактический остаток на начало промо (последнее известное значение)
    actual_stock = 0.0
    for item in request.items:
        promo_start_date = pd.to_datetime(item.promo_start)
        sku_history = historical_df[historical_df['sku'] == item.sku].copy()
        if len(sku_history) > 0:
            # Берем последнее значение перед началом промо
            before_promo = sku_history[sku_history['date'] < promo_start_date]
            if len(before_promo) > 0:
                actual_stock = before_promo.iloc[-1]['sales']
            elif len(sku_history) > 0:
                actual_stock = sku_history.iloc[-1]['sales']
    
    # Заказ = Прогноз - Факт
    order_quantity = max(0, total_forecast - actual_stock)
    
    return ForecastResponse(
        forecast=forecasts,
        model_mape=metadata['test_mape'],
        generated_at=datetime.now().isoformat(),
        total_forecast=round(total_forecast, 2),
        actual_stock=round(actual_stock, 2),
        order_quantity=round(order_quantity, 2)
    )

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "error", "message": "Модель не загружена"}
    return {"status": "ok", "model_mape": metadata['test_mape']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
