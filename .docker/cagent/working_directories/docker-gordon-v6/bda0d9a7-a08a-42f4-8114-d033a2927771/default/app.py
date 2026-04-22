import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://promo-api:8000"

st.set_page_config(page_title="Прогноз промо-продаж", layout="wide")

st.title("📊 Прогнозирование промо-продаж")

# Боковая панель
st.sidebar.header("Параметры промо")

# Ввод товаров
sku = st.sidebar.text_input("SKU товара", "SKU001")
promo_start = st.sidebar.date_input("Начало промо", datetime.now())
promo_end = st.sidebar.date_input("Конец промо", datetime.now() + timedelta(days=3))
discount = st.sidebar.slider("Скидка (%)", 5, 50, 20)
forecast_days = st.sidebar.number_input("Дней прогноза", 7, 30, 14)

if st.sidebar.button("Рассчитать прогноз"):
    # Формируем запрос
    request_data = {
        "items": [
            {
                "sku": sku,
                "promo_start": promo_start.strftime("%Y-%m-%d"),
                "promo_end": promo_end.strftime("%Y-%m-%d"),
                "discount": discount
            }
        ],
        "forecast_days": forecast_days
    }
    
    try:
        response = requests.post(f"{API_URL}/forecast", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            forecast_df = pd.DataFrame(data['forecast'])
            
            # Отображаем метрики
            col1, col2 = st.columns(2)
            col1.metric("MAPE модели", f"{data['model_mape']:.2f}%")
            total_sales = forecast_df['predicted_sales'].sum()
            col2.metric("Прогноз продаж (шт)", f"{total_sales:,.0f}")
            
            # График
            fig = px.line(forecast_df, x='date', y='predicted_sales', 
                         title='Прогноз продаж', markers=True)
            
            # Добавляем период промо
            fig.add_vrect(x0=promo_start.strftime("%Y-%m-%d"), 
                         x1=promo_end.strftime("%Y-%m-%d"),
                         annotation_text="Промо", 
                         annotation_position="top left",
                         fillcolor="green", opacity=0.2, layer="below")
            
            fig.update_layout(xaxis_title="Дата", yaxis_title="Продажи (шт)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Таблица
            st.subheader("Детальный прогноз")
            st.dataframe(forecast_df, use_container_width=True)
            
            # Скачивание
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name=f"forecast_{sku}.csv",
                mime="text/csv"
            )
            
        else:
            st.error(f"Ошибка API: {response.status_code}")
            st.json(response.json())
            
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к API. Убедитесь, что api.py запущен.")
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")

# Информация о модели
with st.expander("📚 Информация о модели"):
    st.write("""
    **Модель:** CatBoost Regressor
    
    **Признаки:**
    - Промо-флаги и скидки
    - Временные признаки (день недели, месяц, выходные)
    - Лаги продаж (1, 7, 14, 28 дней)
    - Скользящие средние и стандартные отклонения
    
    **Метрики качества:**
    - MAPE (средняя абсолютная процентная ошибка)
    - MAE (средняя абсолютная ошибка)
    """)

# Загрузка исторических данных
if st.checkbox("Показать исторические данные"):
    try:
        hist_df = pd.read_csv('sales_history.csv')
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        
        sku_hist = hist_df[hist_df['sku'] == sku]
        
        fig_hist = px.line(sku_hist, x='date', y='sales', 
                          title=f'История продаж {sku}')
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.dataframe(sku_hist, use_container_width=True)
    except Exception as e:
        st.error(f"Не удалось загрузить данные: {str(e)}")
