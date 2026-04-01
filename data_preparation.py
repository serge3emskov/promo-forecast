import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """Создаем тестовые данные для примера"""
    dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
    skus = ['SKU001', 'SKU002', 'SKU003']
    
    data = []
    for sku in skus:
        for date in dates:
            # Базовые продажи
            base_sales = np.random.randint(50, 100)
            
            # Сезонность
            if date.month in [12, 1]:
                base_sales *= 1.3
            elif date.month in [6, 7]:
                base_sales *= 0.8
            
            # День недели
            if date.weekday() >= 5:  # выходные
                base_sales *= 1.2
            
            # Промо
            is_promo = 0
            discount = 0
            # Каждые 2 недели - промо на 3 дня
            if (date.isocalendar()[1] % 2 == 0) and (date.weekday() in [3, 4, 5]):
                is_promo = 1
                discount = np.random.choice([10, 15, 20, 25])
                base_sales *= (1 + discount/100 * 2)  # промо-лифт
            
            sales = int(base_sales + np.random.randint(-10, 10))
            
            data.append({
                'date': date,
                'sku': sku,
                'sales': max(0, sales),
                'is_promo': is_promo,
                'discount': discount if is_promo else 0,
                'price': 100 * (1 - discount/100) if is_promo else 100
            })
    
    df = pd.DataFrame(data)
    df.to_csv('sales_history.csv', index=False)
    print(f"Создано {len(df)} записей")
    return df

if __name__ == "__main__":
    create_sample_data()
