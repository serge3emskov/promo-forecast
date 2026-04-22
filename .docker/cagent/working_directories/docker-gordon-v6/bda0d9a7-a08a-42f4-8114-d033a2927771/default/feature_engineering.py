import pandas as pd
import numpy as np
from datetime import timedelta

def add_features(df):
    """Добавляем признаки для модели"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['sku', 'date']).reset_index(drop=True)
    
    # Временные признаки
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Праздники (упрощенно)
    df['is_holiday'] = ((df['month'] == 1) & (df['day_of_month'].isin([1, 7])) |
                        (df['month'] == 3) & (df['day_of_month'] == 8) |
                        (df['month'] == 5) & (df['day_of_month'] == 1) |
                        (df['month'] == 12) & (df['day_of_month'].isin([31, 1]))).astype(int)
    
    # Лаги (продажи в прошлые периоды)
    for sku in df['sku'].unique():
        mask = df['sku'] == sku
        df.loc[mask, 'lag_1'] = df.loc[mask, 'sales'].shift(1)
        df.loc[mask, 'lag_7'] = df.loc[mask, 'sales'].shift(7)
        df.loc[mask, 'lag_14'] = df.loc[mask, 'sales'].shift(14)
        df.loc[mask, 'lag_28'] = df.loc[mask, 'sales'].shift(28)
        
        # Rolling statistics
        df.loc[mask, 'rolling_mean_7'] = df.loc[mask, 'sales'].shift(1).rolling(7).mean()
        df.loc[mask, 'rolling_mean_14'] = df.loc[mask, 'sales'].shift(1).rolling(14).mean()
        df.loc[mask, 'rolling_std_7'] = df.loc[mask, 'sales'].shift(1).rolling(7).std()
    
    # Промо признаки
    df['days_since_promo'] = df.groupby('sku').apply(
        lambda x: (x['is_promo'] == 0).cumsum() % 14
    ).reset_index(level=0, drop=True)
    
    # Скользящее среднее промо
    for sku in df['sku'].unique():
        mask = df['sku'] == sku
        df.loc[mask, 'promo_freq_28'] = df.loc[mask, 'is_promo'].shift(1).rolling(28).mean()
    
    # Заполняем пропуски
    df = df.fillna(0)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('sales_history.csv')
    df = add_features(df)
    df.to_csv('sales_with_features.csv', index=False)
    print(f"Признаки добавлены. Всего колонок: {len(df.columns)}")
    print(df.head())
