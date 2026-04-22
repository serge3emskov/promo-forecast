# Прогнозирование промо-продаж

FastAPI + Streamlit приложение для прогнозирования объемов продаж во время промо-акций с использованием CatBoost.

## Возможности

- **API (FastAPI)** на порте 8000 — REST API для прогноза продаж
- **UI (Streamlit)** на порте 8501 — интерактивный интерфейс для расчетов
- **ML Pipeline** — автоматическое обучение модели при запуске контейнера
- **Docker Compose** — одной командой запускаются оба сервиса

## Требования

- Docker и Docker Compose
- Git

## Быстрый старт

```bash
git clone <repository-url>
cd promo-forecast
docker compose up -d
```

Затем откройте в браузере:
- **API**: http://localhost:8000/health (проверка здоровья)
- **UI**: http://localhost:8501 (интерфейс пользователя)

## Структура проекта

```
.
├── api.py                    # FastAPI приложение с эндпойнтом /forecast
├── app.py                    # Streamlit интерфейс
├── data_preparation.py       # Создание тестовых данных (sales_history.csv)
├── feature_engineering.py    # Добавление признаков для модели
├── train_model.py            # Обучение CatBoost модели
├── requirements.txt          # Python зависимости
├── Dockerfile                # Docker образ приложения
├── docker-compose.yml        # Оркестрация сервисов
└── .gitignore               # Git исключения
```

## API Endpoints

### POST /forecast
Прогноз продаж для промо-акций

**Запрос:**
```json
{
  "items": [
    {
      "sku": "SKU001",
      "promo_start": "2025-04-01",
      "promo_end": "2025-04-05",
      "discount": 20
    }
  ],
  "forecast_days": 14
}
```

**Ответ:**
```json
{
  "forecast": [
    {
      "sku": "SKU001",
      "date": "2025-04-01",
      "predicted_sales": 125.45,
      "is_promo": true,
      "discount": 20
    }
  ],
  "model_mape": 19.64,
  "generated_at": "2025-04-01T12:00:00"
}
```

### GET /health
Проверка статуса API и модели

**Ответ:**
```json
{
  "status": "ok",
  "model_mape": 19.64
}
```

## Использование

### Через Streamlit UI (http://localhost:8501)

1. Введите SKU товара
2. Выберите период промо (начало и конец)
3. Установите размер скидки (5-50%)
4. Нажмите "Рассчитать прогноз"
5. Скачайте результаты в CSV

### Через API

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

## Логи

```bash
# API логи
docker compose logs -f api

# Streamlit логи
docker compose logs -f streamlit

# Все логи
docker compose logs -f
```

## Остановка приложения

```bash
docker compose down
```

## Технологический стек

- **FastAPI** — REST API framework
- **Streamlit** — UI framework
- **CatBoost** — машинное обучение
- **Pandas** — обработка данных
- **Scikit-learn** — метрики и валидация
- **Docker** — контейнеризация

## Процесс обучения модели

При каждом запуске контейнера автоматически:

1. Генерируются тестовые данные (3 года × 3 SKU)
2. Добавляются признаки (лаги, скользящие средние, временные признаки)
3. Обучается CatBoost модель с кросс-валидацией (3 фолда)
4. Вычисляются метрики (MAPE, MAE)
5. Модель сохраняется в `promo_model.pkl`
6. Стартует API сервер

## Примечания

- Модель переобучается при каждом запуске контейнера (для продакшена сохраняйте модель в volume)
- Исторические данные генерируются синтетически (используйте real data в продакшене)
- MAPE модели ~19-20% на тестовом наборе
