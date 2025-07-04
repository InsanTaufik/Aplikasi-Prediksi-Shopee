# 📊 Sistem Prediksi Penjualan Rumah Bayitaz

## 🎯 Deskripsi Project

Aplikasi berbasis web menggunakan Streamlit untuk memprediksi penjualan produk Rumah Bayitaz selama 12 bulan ke depan. Sistem ini menggunakan machine learning XGBoost dengan fitur-fitur canggih seperti prediksi iteratif, analisis musiman, dan lag features yang telah dikembangkan berdasarkan analisis mendalam dari notebook Jupyter.

## 🚀 Fitur Utama

- **� Dashboard Beranda**: Ringkasan sistem dan statistik utama dalam bahasa Indonesia
- **📈 Prediksi Produk Individual**: Prediksi detail 12 bulan untuk produk tertentu
- **📊 Analisis Keseluruhan**: Prediksi multiple produk sekaligus dengan visualisasi agregat
- **🔍 Eksplorasi Data**: Analisis mendalam data historis dan pola penjualan
- **📱 UI Responsif**: Interface bahasa Indonesia yang user-friendly
- **📥 Export Data**: Download hasil prediksi dalam format CSV

## ⚡ Quick Start

### Windows (Mudah)
```bash
# Double-click file ini:
run_app.bat
```

### Manual
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
python run_streamlit.py

# Atau langsung dengan streamlit
streamlit run Src/app.py
```

Aplikasi akan terbuka di browser di alamat: http://localhost:8501

## 📋 Requirements

- Python 3.8+
- Dependencies dalam `requirements.txt`
- Data penjualan historis dalam folder `Data/`
- Model terlatih dalam folder `saved_model/`

## 🏗️ Arsitektur Sistem

```
Data Historis → Feature Engineering → XGBoost Model → Prediksi Iteratif → Visualisasi
     ↓                ↓                 ↓              ↓              ↓
  CSV Files      Temporal Features  Trained Model    12 Months     Streamlit UI
               Lag Features (1-3M)  Best Params     Forecasting    Interactive Charts
               Rolling Average      Saved .pkl      Dynamic Update  Download CSV
               Seasonal Index       Selected Feat.  Seasonal Adj.   Bahasa Indonesia
```

## 🤖 Model Machine Learning

Sistem menggunakan **XGBoost Regressor** yang telah dioptimasi dengan:

### Fitur-Fitur Utama:
- **Temporal**: Year, Month, Quarter, DaysinMonth, Month_Sin/Cos
- **Lag Features**: Data 1-3 bulan sebelumnya (Total_Jumlah, Penjualan, Views)
- **Rolling Features**: Rata-rata bergulir 3 bulan
- **Seasonal**: Indeks musiman dan tren
- **Product-Specific**: Mean per produk, growth rate, stock features

### Metode Prediksi:
- **Iteratif**: Prediksi bulan demi bulan dengan update fitur dinamis
- **Seasonal Adjustment**: Penyesuaian pola musiman per produk
- **Feature Update**: Lag dan rolling features diperbarui setiap prediksi

### Evaluasi Model:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error
- **R² Score**: Coefficient of Determination

## 📁 Required Input Files
1. `hari_libur_2023-2024.csv` - Holiday dates data
2. `Penjualan Rumah Bayita Shopee 2023-2024 (FULL).csv` - Main sales data
3. `Pesanan Siap Dikirim 2023-2024.csv` - Ready-to-ship orders data
4. `Status Stok Penjualan 2023-2024 RumahBayitaz.csv` - Stock status data
5. `stok_toko.csv` - Store stock data

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Demo Model (if you don't have the trained model yet)
```bash
cd Src
python create_demo_model.py
```

### 3. Run the Streamlit App

#### Option 1: Using the Launcher Scripts (Recommended)
**Windows:**
```cmd
run_app.bat
```

**Python (All platforms):**
```bash
python run_app.py
```

#### Option 2: Direct Command
```bash
streamlit run streamlit_app_complete.py --server.port 8503
```

## 📌 App Features

### 🔄 Data Processing
- **Automatic File Upload**: Upload interface for all 5 required CSV files
- **Data Cleaning**: Handles missing values, date parsing, and data type conversion
- **Feature Engineering**: Creates 20+ features including:
  - Temporal features (Year, Month, Quarter, cyclical encoding)
  - Lag features (1, 2, 3 months)
  - Rolling averages
  - Growth rates
  - Product-specific statistics
  - Holiday indicators
  - Price and efficiency ratios

### 🔮 Predictions
- **XGBoost Model**: Uses pre-trained XGBoost regressor
- **Multi-Product Forecasting**: Generates predictions for top-performing products
- **Growth Analysis**: Calculates growth percentages
- **Validation**: Ensures non-negative predictions

### 📊 Visualizations
- **Interactive Charts**: Built with Plotly
- **Comparison Views**: Current vs predicted sales
- **Growth Rate Analysis**: Visual growth indicators
- **Time Series Plots**: Historical trends
- **Responsive Design**: Works on different screen sizes

### 💾 Export Features
- **CSV Download**: Export predictions and processed data
- **Timestamped Files**: Automatic file naming with timestamps
- **Multiple Formats**: Both summary and detailed exports

## 🎨 UI Features
- **Modern Design**: Gradient backgrounds and card layouts
- **Responsive Layout**: Wide layout with sidebar navigation
- **Progress Indicators**: Loading spinners and progress messages
- **Color-coded Results**: Visual indicators for performance
- **Tabbed Interface**: Organized content in tabs

## 🔧 Technical Details

### Data Preprocessing Pipeline
1. **Load and Parse**: Read CSV files with proper date parsing
2. **Aggregate**: Group by product and month
3. **Feature Creation**: Generate time-series and business features
4. **External Data Integration**: Merge with stock and shipping data
5. **Missing Value Handling**: Fill NaN values appropriately

### Model Integration
- **Joblib Loading**: Efficient model serialization
- **Feature Alignment**: Ensures correct feature order and presence
- **Error Handling**: Graceful error handling for missing features
- **Batch Prediction**: Processes multiple products efficiently

### File Structure
```
├── streamlit_app_complete.py     # Main comprehensive Streamlit application
├── run_app.py                    # Python launcher script
├── run_app.bat                   # Windows batch launcher
├── test_app.py                   # Test script for validation
├── Src/
│   ├── streamlit_app.py          # Original Streamlit application
│   └── create_demo_model.py      # Demo model creation script
├── saved_model/
│   ├── xgboost_regressor_TUNED.pkl    # Trained XGBoost model
│   └── selected_features.pkl          # List of selected features
├── Data/                         # Upload your CSV files here
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## 🔄 Usage Workflow

1. **Start the App**: 
   - **Windows**: Double-click `run_app.bat` or run `python run_app.py`
   - **Other platforms**: Run `python run_app.py` or `streamlit run streamlit_app_complete.py`
2. **Upload Files**: Use the sidebar to upload all 5 CSV files
3. **Process Data**: Click "Process Data & Generate Predictions"
4. **View Results**: Explore the different tabs:
   - **Data Overview**: See processed data statistics
   - **Predictions**: View prediction results table
   - **Visualizations**: Interactive charts and graphs
   - **Download Results**: Export CSV files
5. **Download**: Export results for further analysis

## 🎯 Performance Features
- **Top Products Focus**: Analyzes top-performing products for better insights
- **Efficient Processing**: Optimized for large datasets
- **Memory Management**: Handles data efficiently
- **Fast Predictions**: Quick model inference

## 🚨 Troubleshooting

### Common Issues
1. **Model Not Found**: Run `create_demo_model.py` to create a demo model
2. **Import Errors**: Install all dependencies with `pip install -r requirements.txt`
3. **CSV Format Issues**: Ensure CSV files have the expected columns
4. **Memory Issues**: Try with smaller datasets first

### Requirements
- Python 3.8+
- 8GB+ RAM recommended for large datasets
- Modern web browser

## 🔮 Future Enhancements
- **Real-time Data**: Connect to live data sources
- **Model Retraining**: Interface for updating the model
- **Advanced Analytics**: More sophisticated forecasting
- **API Integration**: REST API for programmatic access
- **Multi-language Support**: Internationalization
- **Mobile Optimization**: Better mobile experience

## 📝 License
This project is for educational and research purposes.
