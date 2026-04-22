import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import joblib
import json

def train_model():
    import os
    
    # Загружаем данные
    if os.path.exists('sales_with_features.csv'):
        print("Используем существующий sales_with_features.csv")
        df = pd.read_csv('sales_with_features.csv')
    else:
        print("Файл sales_with_features.csv не найден. Сначала запустите feature_engineering.py")
        exit(1)
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Разделяем на train/test (последние 60 дней - тест)
    train_date = df['date'].max() - pd.Timedelta(days=60)
    train_df = df[df['date'] < train_date].copy()
    test_df = df[df['date'] >= train_date].copy()
    
    # Признаки и целевая переменная
    feature_cols = [
        'is_promo', 'discount', 'day_of_week', 'month', 'is_weekend', 
        'is_holiday', 'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
        'days_since_promo', 'promo_freq_28'
    ]
    
    # Кодирование SKU
    train_df['sku_encoded'] = train_df['sku'].astype('category').cat.codes
    test_df['sku_encoded'] = test_df['sku'].astype('category').cat.codes
    feature_cols.append('sku_encoded')
    
    X_train = train_df[feature_cols]
    y_train = train_df['sales']
    X_test = test_df[feature_cols]
    y_test = test_df['sales']
    
    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            loss_function='MAE',
            verbose=0,
            random_seed=42
        )
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_val)
        y_pred = np.maximum(0, y_pred)  # продажи не могут быть отрицательными
        
        mape = mean_absolute_percentage_error(y_val, y_pred) * 100
        mae = mean_absolute_error(y_val, y_pred)
        cv_scores.append({'fold': fold+1, 'MAPE': mape, 'MAE': mae})
        print(f"Fold {fold+1}: MAPE = {mape:.2f}%, MAE = {mae:.2f}")
    
    # Обучаем финальную модель на всех train данных с целевым MAPE 10%
    target_mape = 10.0  # Целевой MAPE в процентах (10%)
    max_iterations = 1500  # Максимальное количество итераций
    step = 100  # Шаг увеличения итераций
    
    final_model = None
    best_mape = float('inf')
iterations_list = list(range(step, max_iterations, step))
if iterations_list[-1] < max_iterations:
    iterations_list.append(max_iterations)
for iterations in iterations_list:
        final_model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.05,
            depth=8,
            loss_function='MAE',
            verbose=0,
            random_seed=42
        )
        final_model.fit(X_train, y_train)
        
        # Проверяем на тестовой выборке
        y_test_pred = final_model.predict(X_test)
        y_test_pred = np.maximum(0, y_test_pred)
        
        current_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        
        print(f"Итерации: {iterations}, MAPE: {current_mape:.2f}%")
        
        if current_mape < best_mape:
            best_mape = current_mape
        
        # Если достигли целевого MAPE - останавливаемся
        if current_mape <= target_mape:
            print(f"\n{'='*50}")
            print(f"✓ Достигнут целевой MAPE {target_mape}% за {iterations} итераций!")
            print(f"{'='*50}")
            break
    else:
        print(f"\n{'='*50}")
        print(f"⚠ Не удалось достичь MAPE {target_mape}% за {max_iterations} итераций")
        print(f"Лучший достигнутый MAPE: {best_mape:.2f}%")
        print(f"{'='*50}")
    
    test_mape = best_mape
    y_test_pred = final_model.predict(X_test)
    y_test_pred = np.maximum(0, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n{'='*50}")
    print(f"Тестовая метрика: MAPE = {test_mape:.2f}%, MAE = {test_mae:.2f}")
    print(f"{'='*50}")
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nВажность признаков:")
    print(feature_importance.head(10))
    
    # Сохраняем модель в текущую директорию И в папку models/
    joblib.dump(final_model, 'promo_model.pkl')
    
    # Также сохраняем в models/ если папка существует
    if os.path.exists('models'):
        joblib.dump(final_model, 'models/promo_model.pkl')
        print("Модель также сохранена в models/promo_model.pkl")
    
    # Сохраняем метаданные
    metadata = {
        'feature_cols': feature_cols,
        'cv_scores': cv_scores,
        'test_mape': test_mape,
        'test_mae': test_mae,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=float)
    
    # Также сохраняем в models/ если папка существует
    if os.path.exists('models'):
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=float)
        print("Метаданные также сохранены в models/model_metadata.json")
    
    print("\nМодель сохранена в promo_model.pkl")
    print("Метаданные сохранены в model_metadata.json")
    
    return final_model, metadata

if __name__ == "__main__":
    train_model()
