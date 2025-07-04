@echo off
echo ========================================
echo    Rumah Bayitaz Sales Prediction App
echo ========================================
echo.
echo Memulai aplikasi prediksi penjualan...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python tidak ditemukan!
    echo Pastikan Python sudah terinstall dan ada di PATH
    pause
    exit /b 1
)

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: File requirements.txt tidak ditemukan!
    pause
    exit /b 1
)

REM Check if the main app exists
if not exist "Src\app_final_bahasa_indonesia.py" (
    echo ERROR: File aplikasi utama tidak ditemukan!
    echo Pastikan file Src\app_final_bahasa_indonesia.py ada
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Gagal menginstall beberapa dependencies
    echo Mencoba menjalankan aplikasi...
    echo.
)

REM Create Data folder if it doesn't exist
if not exist "Data" (
    echo Membuat folder Data...
    mkdir Data
)

REM Start the Streamlit app
echo.
echo Menjalankan aplikasi Streamlit...
echo Aplikasi akan terbuka di browser secara otomatis
echo URL: http://localhost:8501
echo.
echo Tekan Ctrl+C untuk menghentikan aplikasi
echo.

streamlit run "Src\app_final_bahasa_indonesia.py" --server.port 8501 --server.address localhost

REM If streamlit command fails, try with python -m streamlit
if %errorlevel% neq 0 (
    echo.
    echo Mencoba dengan python -m streamlit...
    python -m streamlit run "Src\app_final_bahasa_indonesia.py" --server.port 8501 --server.address localhost
)

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Gagal menjalankan aplikasi!
    echo Pastikan Streamlit sudah terinstall dengan: pip install streamlit
    pause
    exit /b 1
)

echo.
echo Aplikasi telah berhenti.
pause
