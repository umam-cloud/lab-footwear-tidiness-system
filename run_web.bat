@echo off
title Sistem Informasi Klasifikasi Kerapian Sandal Lab - PSD
echo =====================================================================
echo    SISTEM INFORMASI KLASIFIKASI KERAPIAN SANDAL LABORATORIUM (PSD)
echo               Model: XGBoost Classifier (Akurasi: 97.44%)
echo =====================================================================
echo.
echo Menjalankan server Flask...
start http://localhost:5000
python app.py
pause
