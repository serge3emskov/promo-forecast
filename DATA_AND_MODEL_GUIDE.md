# Инструкция по добавлению данных и модели

## Обзор

Проект поддерживает два режима работы:
1. **Синтетические данные** - автоматически генерируются при запуске
2. **Ваши собственные данные** - можно загрузить свои исторические данные и модель

## Способ 1: Использование собственных данных

### Шаг 1: Подготовка файла с данными

Создайте файл `data/sales_history.csv` со следующими колонками:

| Колонка | Тип | Описание |
|---------|-----|----------|
| date | date | Дата продажи (YYYY-MM-DD) |
| sku | string | Артикул товара |
| sales | int | Количество продаж |
| is_promo | int | Флаг промо-акции (0 или 1) |
| discount | float | Размер скидки в процентах (0-50) |
| price | float | Цена товара |

**Пример данных:**
```csv
date,sku,sales,is_promo,discount,price
2023-01-01,SKU001,85,0,0,100
2023-01-02,SKU001,92,0,0,100
2023-01-03,SKU001,78,1,15,85
2023-01-04,SKU001,110,1,15,85
```

### Шаг 2: Запуск с вашими данными

```bash
# Поместите ваш файл в папку data/
# data/sales_history.csv уже будет использован автоматически

docker compose up -d
```

Система автоматически:
- Обнаружит файл `sales_history.csv`
- Пропустит генерацию синтетических данных
- Добавит признаки (feature engineering)
- Обучит модель (если нет готовой)

## Способ 2: Использование собственной обученной модели

### Шаг 1: Сохранение модели

Сохраните вашу CatBoost модель в формате pickle:

```python
import joblib
joblib.dump(your_model, 'models/promo_model.pkl')
```

Также создайте файл `models/model_metadata.json`:

```json
{
  "feature_cols": [
    "is_promo", "discount", "day_of_week", "month", "is_weekend",
    "is_holiday", "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "days_since_promo", "promo_freq_28", "sku_encoded"
  ],
  "test_mape": 15.5
}
```

### Шаг 2: Запуск с готовой моделью

```bash
# Поместите файлы в папки:
# models/promo_model.pkl
# models/model_metadata.json (опционально)

docker compose up -d
```

Система обнаружит модель и пропустит этап обучения.

## Способ 3: Полностью своя конфигурация

Если вы хотите использовать свои данные И свою модель:

1. Положите `sales_history.csv` в папку `data/`
2. Положите `promo_model.pkl` в папку `models/`
3. Запустите:

```bash
docker compose up -d
```

## Структура папок после добавления

```
/workspace
├── data/
│   └── sales_history.csv      # Ваши исторические данные
├── models/
│   ├── promo_model.pkl        # Обученная модель
│   └── model_metadata.json    # Метаданные модели
├── docker-compose.yml
├── Dockerfile
└── ... (остальные файлы проекта)
```

## Проверка работы

После запуска проверьте логи:

```bash
# Логи API
docker compose logs -f api

# Логи Streamlit
docker compose logs -f streamlit
```

Вы должны увидеть сообщения:
- "Using existing sales_history.csv" (если используете свои данные)
- "Using existing model" (если используете свою модель)

## API Endpoints

- **API**: http://localhost:8000/health
- **UI**: http://localhost:8501

## Пример запроса к API

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "sku": "SKU001",
        "promo_start": "2025-04-01",
        "promo_end": "2025-04-05",
        "discount": 20
      }
    ],
    "forecast_days": 14
  }'
```

## Требования к данным

Для корректной работы модели убедитесь, что:

1. **Минимум 6 месяцев** исторических данных
2. **Ежедневные данные** без пропусков
3. **Минимум 3 SKU** для лучшей работы модели
4. **Промо-периоды** отмечены флагом is_promo=1
5. **Скидки** указаны в процентах (0-50)

## Troubleshooting

### Модель не загружается
Убедитесь, что файл `promo_model.pkl` находится в папке `models/` и имеет правильные права доступа.

### Данные не распознаются
Проверьте формат дат (YYYY-MM-DD) и названия колонок в CSV файле.

### Ошибки при обучении
Запустите контейнер в режиме отладки:
```bash
docker compose up --build
```
