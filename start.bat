@echo off
cd /d C:\Users\serge\promo-forecast
docker compose up -d
echo Приложение запущено!
echo API: http://localhost:8000
echo UI: http://localhost:8501
pause

