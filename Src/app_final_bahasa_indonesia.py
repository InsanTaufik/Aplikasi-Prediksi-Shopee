"""
SISTEM PREDIKSI PENJUALAN RUMAH BAYITAZ - BAHASA INDONESIA
===========================================================

Aplikasi Streamlit untuk prediksi penjualan dengan menggunakan XGBoost
yang diimplementasikan sesuai dengan notebook Jupyter.

Fitur utama:
- Upload dan preprocessing data otomatis
- Prediksi iteratif sesuai notebook
- Visualisasi hasil prediksi
- Analisis performa model
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
import sys
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from typing import Optional

# Add current directory to path for imports
sys.path.append('.')
sys.path.append('Src')

# Import core functions (production-ready backend)
try:
    from core_functions import process_uploaded_data_core, create_features, train_model, iterative_forecast
    from data_preprocessing import DataPreprocessor, process_data_to_xgboost_ready
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"⚠️ Backend tidak tersedia: {e}")

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Penjualan Rumah Bayitaz",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling yang modern dan profesional
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-bottom: 1rem;
        font-weight: 600;
        text-align: center;
        font-style: italic;
    }
    .metric-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
        color: #333333;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        color: #155724;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        color: #856404;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
        color: #721c24;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
        color: #0c5460;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: #333;
    }
    .algorithm-info {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #333;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .performance-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .performance-metric h3 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .performance-metric p {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .main-section {
        background-color: transparent;
        padding: 1rem;
        border-radius: 0px;
        margin-bottom: 1rem;
        box-shadow: none;
        border: none;
    }
    .section-title {
        font-size: 2rem;
        color: #1f77b4;
        margin-bottom: 1.5rem;
        font-weight: bold;
        text-align: center;
        border-bottom: none;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Logger untuk error handling yang robust
class ProductionErrorLogger:
    def __init__(self):
        self.errors = []
        self.processing_log = []
    
    def log_error(self, step: str, error: Exception, context: Optional[dict] = None):
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context if context is not None else {}
        }
        self.errors.append(error_entry)
        st.error(f"❌ Error dalam {step}: {str(error)}")
        
        with st.expander("🔍 Detail Error"):
            st.json(error_entry)
    
    def log_info(self, step: str, message: str, data_info: Optional[dict] = None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'message': message,
            'data_info': data_info or {}
        }
        self.processing_log.append(log_entry)
        st.info(f"ℹ️ {step}: {message}")
    
    def get_summary(self):
        return {
            'total_errors': len(self.errors),
            'total_logs': len(self.processing_log),
            'recent_errors': self.errors[-3:] if self.errors else [],
            'processing_summary': self.processing_log[-5:] if self.processing_log else []
        }

# Initialize logger
logger = ProductionErrorLogger()

# Fitur yang digunakan dalam model (sesuai notebook)
SELECTED_FEATURES = [
    'Rolling_3M_Total_Jumlah',
    'Mean_Total_Jumlah_per_Product', 
    'Lag_1_Total_Jumlah',
    'Seasonal_Index',
    'Trend_Total_Jumlah',
    'Lag_2_Total_Jumlah',
    'MoM_Growth_Total_Jumlah',
    'Lag_3_Total_Jumlah',
    'Total_Pesanan'
]

# Fungsi untuk membaca CSV dengan error handling
def safe_read_csv(uploaded_file, encoding='utf-8'):
    """Membaca file CSV dengan error handling yang robust"""
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
        logger.log_info("Pembacaan File", f"Berhasil memuat {uploaded_file.name}", 
                        {'shape': df.shape, 'columns': list(df.columns)})
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
            logger.log_info("Pembacaan File", f"Berhasil memuat {uploaded_file.name} dengan encoding latin1")
            return df
        except Exception as e:
            logger.log_error("Pembacaan File", e, {'file': uploaded_file.name, 'encoding': 'latin1'})
            return None
    except Exception as e:
        logger.log_error("Pembacaan File", e, {'file': uploaded_file.name})
        return None

# Helper functions untuk visualisasi dan forecasting
def calculate_seasonal_index(product_data):
    """Hitung indeks musiman untuk setiap bulan berdasarkan data historis"""
    product_data = product_data.copy()
    product_data['Month'] = product_data['Bulan'].dt.month
    
    # Hitung rata-rata bulanan
    monthly_avg = product_data.groupby('Month')['Total_Jumlah'].mean()
    overall_avg = product_data['Total_Jumlah'].mean()
    
    # Hitung indeks musiman untuk setiap bulan
    seasonal_indices = {}
    for month in range(1, 13):
        if month in monthly_avg.index and overall_avg > 0:
            seasonal_indices[month] = monthly_avg[month] / overall_avg
        else:
            seasonal_indices[month] = 1.0
    
    return seasonal_indices

def calculate_trend(values, window=3):
    """Hitung tren sederhana dari nilai-nilai terbaru"""
    if len(values) < 2:
        return 0.0
    
    # Gunakan regresi linear pada nilai terbaru
    recent_values = values[-window:]
    if len(recent_values) < 2:
        return 0.0
    
    x = np.arange(len(recent_values))
    slope = np.polyfit(x, recent_values, 1)[0]
    return max(0, slope)  # Pastikan tren non-negatif

def create_forecast_visualization(historical_data, forecast_data, product_name):
    """Buat visualisasi perbandingan data historis dan prediksi"""
    fig = go.Figure()
    
    # Data historis
    fig.add_trace(go.Scatter(
        x=historical_data['Bulan'],
        y=historical_data['Total_Jumlah'],
        mode='lines+markers',
        name='Data Historis',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Data prediksi
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(forecast_data['Bulan']),
        y=forecast_data['Prediksi'],
        mode='lines+markers',
        name='Prediksi',
        line=dict(color='orange', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title=f'📈 Prediksi Penjualan: {product_name[:60]}...',
        xaxis_title='Tanggal',
        yaxis_title='Total Jumlah',
        hovermode='x unified',
        height=500,
        showlegend=True,
        font=dict(size=12)
    )
    
    return fig

def create_historical_vs_forecast_chart(df, forecast_results, product_name):
    """Buat chart kombinasi data historis dan prediksi seperti app_production_ready.py"""
    try:
        # Get historical data for the product
        product_data = df[df['Produk_Variasi'] == product_name].copy()
        
        if product_data.empty:
            st.warning(f"Tidak ada data historis untuk {product_name}")
            return None
        
        # Sort by month for display
        product_data = product_data.sort_values('Bulan')
        
        # Historical values
        historical_months = product_data['Bulan'].tolist()
        historical_sales = product_data['Total_Jumlah'].tolist()
        
        # Forecast values
        forecast_months = [pd.to_datetime(result['Bulan']) for result in forecast_results]
        forecast_sales = [result['Prediksi'] for result in forecast_results]
        
        # Create the plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_months,
            y=historical_sales,
            mode='lines+markers',
            name='Data Historis',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='lightblue')
        ))
        
        # Future predictions
        fig.add_trace(go.Scatter(
            x=forecast_months,
            y=forecast_sales,
            mode='lines+markers',
            name='Prediksi',
            line=dict(color='orange', dash='dash', width=3),
            marker=dict(size=10, symbol='diamond', color='orange')
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'📊 Perbandingan Data Historis vs Prediksi: {product_name}',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title='Bulan',
            yaxis_title='Jumlah Penjualan',
            hovermode='x unified',
            showlegend=True,
            height=600,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error membuat visualisasi: {str(e)}")
        return None

def show_model_performance():
    """Tampilkan metrik performa model dari notebook"""
    st.markdown("### 📊 Performa Model")
    
    # Metrik performa sebenarnya dari notebook
    performance_data = {
        'MAE': {'value': 0.873, 'label': 'Mean Absolute Error', 'desc': 'Rata-rata kesalahan absolut'},
        'RMSE': {'value': 3.216, 'label': 'Root Mean Square Error', 'desc': 'Akar rata-rata kuadrat kesalahan'},
        'MAPE': {'value': 9.763, 'label': 'Mean Absolute Percentage Error', 'desc': 'Rata-rata persentase kesalahan'},
        'R²': {'value': 0.901, 'label': 'Coefficient of Determination', 'desc': 'Koefisien determinasi (akurasi model)'}
    }
    
    # Buat kartu performa
    cols = st.columns(4)
    for i, (metric, data) in enumerate(performance_data.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="performance-metric">
                <h3>{data['value']}</h3>
                <p><strong>{metric}</strong></p>
                <p>{data['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Penjelasan tambahan
    st.markdown("""
    <div class="info-box">
        <h4>📈 Interpretasi Metrik Performa</h4>
        <ul>
            <li><strong>MAE (0.873):</strong> Model rata-rata meleset 0.873 unit dari nilai sebenarnya</li>
            <li><strong>RMSE (3.216):</strong> Deviasi standar prediksi dari nilai aktual</li>
            <li><strong>MAPE (9.763%):</strong> Rata-rata kesalahan persentase sekitar 9.76%</li>
            <li><strong>R² (0.901):</strong> Model menjelaskan 90.1% variasi dalam data (sangat baik!)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def show_algorithm_info():
    """Tampilkan informasi algoritma yang digunakan"""
    st.markdown("### 🤖 Informasi Algoritma")
    
    st.markdown("""
    <div class="algorithm-info">
        <h4>🎯 XGBoost Regressor (TUNED)</h4>
        <p><strong>Algoritma:</strong> Extreme Gradient Boosting</p>
        <p><strong>Tipe:</strong> Ensemble Learning - Gradient Boosting</p>
        <p><strong>Keunggulan:</strong></p>
        <ul>
            <li>Mampu menangani data non-linear dengan baik</li>
            <li>Tahan terhadap overfitting</li>
            <li>Efisien dalam pemrosesan data besar</li>
            <li>Dapat menangani missing values</li>
        </ul>
        <p><strong>Hyperparameter Tuning:</strong> Telah dioptimalkan menggunakan GridSearchCV</p>
    </div>
    """, unsafe_allow_html=True)

def show_selected_features():
    """Tampilkan fitur yang dipilih untuk model"""
    st.markdown("### 🎯 Fitur yang Digunakan Model")
    
    feature_descriptions = {
        'Rolling_3M_Total_Jumlah': 'Rata-rata penjualan 3 bulan terakhir',
        'Mean_Total_Jumlah_per_Product': 'Rata-rata penjualan historis produk',
        'Lag_1_Total_Jumlah': 'Penjualan bulan sebelumnya',
        'Seasonal_Index': 'Indeks musiman berdasarkan bulan',
        'Trend_Total_Jumlah': 'Tren penjualan terkini',
        'Lag_2_Total_Jumlah': 'Penjualan 2 bulan sebelumnya',
        'MoM_Growth_Total_Jumlah': 'Pertumbuhan month-over-month',
        'Lag_3_Total_Jumlah': 'Penjualan 3 bulan sebelumnya',
        'Total_Pesanan': 'Total pesanan pada periode yang sama'
    }
    
    st.markdown("""
    <div class="feature-card">
        <h4>📋 Daftar Fitur Terpilih (9 Fitur)</h4>
    </div>
    """, unsafe_allow_html=True)
    
    for i, feature in enumerate(SELECTED_FEATURES, 1):
        description = feature_descriptions.get(feature, 'Deskripsi tidak tersedia')
        st.markdown(f"**{i}. {feature}**")
        st.markdown(f"   _{description}_")

def show_data_summary(data):
    """Tampilkan ringkasan data yang komprehensif"""
    st.markdown("### 📊 Ringkasan Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{data.shape[0]:,}</h3>
            <p>Total Baris Data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{data['Produk_Variasi'].nunique():,}</h3>
            <p>Produk Unik</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        date_range = (data['Bulan'].max() - data['Bulan'].min()).days // 30
        st.markdown(f"""
        <div class="metric-box">
            <h3>{date_range}</h3>
            <p>Rentang Bulan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_sales = data['Total_Jumlah'].sum()
        st.markdown(f"""
        <div class="metric-box">
            <h3>{total_sales:,.0f}</h3>
            <p>Total Penjualan</p>
        </div>
        """, unsafe_allow_html=True)

# Fungsi untuk forecasting iteratif yang tepat
def iterative_forecast_exact(df, model, features, product, n_months=12):
    """
    Forecasting iteratif sesuai dengan notebook - implementasi yang tepat
    """
    # Dapatkan data historis untuk produk
    product_data = df[df['Produk_Variasi'] == product].copy()
    product_data = product_data.sort_values('Bulan')

    if len(product_data) == 0:
        return []

    # Hitung indeks musiman spesifik produk
    seasonal_indices = calculate_seasonal_index(product_data)

    # Dapatkan nilai historis terbaru untuk inisialisasi
    recent_sales = product_data['Total_Jumlah'].tolist()
    recent_orders = product_data['Total_Pesanan'].tolist()

    # Hitung karakteristik produk yang stabil
    mean_total_jumlah = product_data['Total_Jumlah'].mean()
    mean_total_pesanan = product_data['Total_Pesanan'].mean()
    historical_mom_growth = product_data['MoM_Growth_Total_Jumlah'].mean()

    # Buat tanggal masa depan
    last_date = product_data['Bulan'].max()
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=n_months,
        freq='MS'
    )

    # Inisialisasi hasil prediksi
    forecast_results = []
    predicted_sales = []

    for i, future_date in enumerate(future_dates):
        # Hitung fitur untuk bulan masa depan
        current_month = future_date.month
        
        # Lag features dari nilai sebelumnya
        lag_1 = recent_sales[-1] if len(recent_sales) >= 1 else 0
        lag_2 = recent_sales[-2] if len(recent_sales) >= 2 else 0
        lag_3 = recent_sales[-3] if len(recent_sales) >= 3 else 0
        
        # Rolling 3-month average
        rolling_3m = np.mean(recent_sales[-3:]) if len(recent_sales) >= 3 else mean_total_jumlah
        
        # Seasonal index
        seasonal_index = seasonal_indices.get(current_month, 1.0)
        
        # Trend calculation
        trend = calculate_trend(recent_sales)
        
        # MoM growth (menggunakan rata-rata historis)
        mom_growth = historical_mom_growth
        
        # Total pesanan (menggunakan rata-rata historis)
        total_pesanan = mean_total_pesanan
        
        # Buat array fitur sesuai urutan SELECTED_FEATURES
        feature_values = [
            rolling_3m,                 # Rolling_3M_Total_Jumlah
            mean_total_jumlah,          # Mean_Total_Jumlah_per_Product
            lag_1,                      # Lag_1_Total_Jumlah
            seasonal_index,             # Seasonal_Index
            trend,                      # Trend_Total_Jumlah
            lag_2,                      # Lag_2_Total_Jumlah
            mom_growth,                 # MoM_Growth_Total_Jumlah
            lag_3,                      # Lag_3_Total_Jumlah
            total_pesanan               # Total_Pesanan
        ]
        
        # Prediksi menggunakan model
        try:
            prediction = model.predict([feature_values])[0]
            prediction = max(0, round(prediction))  # Pastikan prediksi non-negatif dan bulat
        except Exception as e:
            logger.log_error("Prediksi", e, {'features': feature_values})
            prediction = round(mean_total_jumlah)  # Fallback ke rata-rata (bulat)
        
        # Simpan hasil (rounded to integer)
        forecast_results.append({
            'Bulan': future_date.strftime('%Y-%m'),
            'Prediksi': int(round(prediction))
        })
        
        # Update riwayat untuk prediksi berikutnya
        recent_sales.append(prediction)
        predicted_sales.append(prediction)
        
        # Batasi panjang riwayat
        if len(recent_sales) > 12:
            recent_sales.pop(0)

    return forecast_results

# Initialize session state untuk menyimpan data
if 'data_processed' not in st.session_state:
    st.session_state['data_processed'] = False
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'features' not in st.session_state:
    st.session_state['features'] = SELECTED_FEATURES

st.markdown('<div class="main-header">🏠 Sistem Prediksi Penjualan Rumah Bayitaz</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">📊 Aplikasi untuk Prediksi Penjualan</div>', unsafe_allow_html=True)

# Sidebar dengan tampilan menarik
st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <h3 style="margin: 0; flex-grow: 1;">🎯 Navigasi Utama</h3>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["📁 Upload & Proses Data", "📈 Prediksi Penjualan", "📊 Analisis Model & Performa"],
    help="Gunakan dropdown untuk navigasi antar fitur aplikasi"
)

# Display sistem status
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔧 Status Sistem")
    if BACKEND_AVAILABLE:
        st.success("✅ Backend Aktif")
    else:
        st.error("❌ Backend Tidak Tersedia")
    
    # Model status check
    model_path = 'saved_model/xgboost_regressor_TUNED.pkl'
    data_path = 'Data/xgboost_ready_dataset.csv'
    features_path = 'saved_model/selected_features.pkl'
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        st.success("✅ Model Tersedia")
        st.session_state['model_loaded'] = True
    else:
        st.warning("⚠️ Model Belum Dilatih")
        st.session_state['model_loaded'] = False
    
    if os.path.exists(data_path):
        # Check if data was processed in current session or file exists
        if st.session_state.get('data_processed', False):
            st.success("✅ Data Siap (Sesi Ini)")
        else:
            st.info("📄 Data Tersedia (Belum Dimuat)")
        st.session_state['data_processed'] = st.session_state.get('data_processed', False)
    else:
        st.warning("⚠️ Data Belum Diproses")
        st.session_state['data_processed'] = False

# Main content based on menu selection
if menu == "📁 Upload & Proses Data":
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📁 Upload & Proses Data</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>📋 Persyaratan File Data</h4>
        <p>Silakan upload <strong>5 file CSV</strong> yang dibutuhkan sesuai format notebook:</p>
        <ul>
            <li><strong>📅 Hari Libur:</strong> File dengan kolom 'Tanggal' format DD-MM-YYYY</li>
            <li><strong>🛒 Penjualan:</strong> File dengan kolom sales transaction data</li>
            <li><strong>📦 Pesanan Siap:</strong> File dengan kolom pesanan yang siap dikirim</li>
            <li><strong>📊 Status Stok:</strong> File dengan kolom status stok per bulan</li>
            <li><strong>🏪 Stok Toko:</strong> File dengan kolom data stok toko</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📤 Upload File Baru", "🔄 Gunakan Data Existing"])
    
    with tab1:
        st.markdown("### 📤 Upload File Data")
        col1, col2 = st.columns(2)
        
        with col1:
            hari_libur_file = st.file_uploader("📅 File Hari Libur", type=['csv'], key='hari_libur', 
                                               help="Upload file CSV dengan data hari libur")
            penjualan_file = st.file_uploader("🛒 File Penjualan", type=['csv'], key='penjualan',
                                              help="Upload file CSV dengan data penjualan")
            pesanan_siap_file = st.file_uploader("📦 File Pesanan Siap", type=['csv'], key='pesanan_siap',
                                                 help="Upload file CSV dengan data pesanan siap")
        
        with col2:
            status_stok_file = st.file_uploader("📊 File Status Stok", type=['csv'], key='status_stok',
                                               help="Upload file CSV dengan data status stok")
            stok_toko_file = st.file_uploader("🏪 File Stok Toko", type=['csv'], key='stok_toko',
                                             help="Upload file CSV dengan data stok toko")

        # Validation and processing
        all_files = [hari_libur_file, penjualan_file, pesanan_siap_file, status_stok_file, stok_toko_file]
        uploaded_count = len([f for f in all_files if f is not None])
        
        # Progress indicator
        st.markdown(f"### 📊 Progress Upload: {uploaded_count}/5 file")
        progress_pct = (uploaded_count / 5) * 100
        st.progress(progress_pct / 100, text=f"{uploaded_count} dari 5 file telah diupload")
        
        if st.button("🚀 Proses Semua Data", type="primary", disabled=(uploaded_count < 5)):
            if uploaded_count == 5 and BACKEND_AVAILABLE:
                
                with st.spinner("🔄 Memproses data menggunakan pipeline notebook..."):
                    try:
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Save uploaded files to Data folder
                        status_text.text("� Menyimpan file yang diupload...")
                        progress_bar.progress(5)
                        time.sleep(0.5)
                        
                        # Define file mapping
                        file_mapping = [
                            (hari_libur_file, 'hari_libur_2023-2024.csv'),
                            (penjualan_file, 'Penjualan Rumah Bayita Shopee 2023-2024 (FULL).csv'),
                            (pesanan_siap_file, 'Pesanan Siap Dikirim 2023-2024.csv'),
                            (status_stok_file, 'Status Stok Penjualan 2023-2024 RumahBayitaz.csv'),
                            (stok_toko_file, 'stok_toko.csv')
                        ]
                        
                        # Save uploaded files
                        saved_files = []
                        
                        # Create Data directory if it doesn't exist
                        data_dir = 'Data'
                        if not os.path.exists(data_dir):
                            os.makedirs(data_dir)
                            st.info(f"📁 Folder {data_dir} berhasil dibuat")
                        
                        for uploaded_file, filename in file_mapping:
                            if uploaded_file is not None:
                                try:
                                    file_path = os.path.join(data_dir, filename)
                                    with open(file_path, 'wb') as f:
                                        f.write(uploaded_file.getbuffer())
                                    saved_files.append(f"✅ {uploaded_file.name} → {filename}")
                                except Exception as save_error:
                                    st.error(f"❌ Gagal menyimpan {uploaded_file.name}: {str(save_error)}")
                                    raise save_error
                        
                        # Show saved files
                        for file_info in saved_files:
                            st.success(file_info)
                        
                        # Step 2: Initialize processor
                        status_text.text("� Inisialisasi DataPreprocessor...")
                        progress_bar.progress(10)
                        time.sleep(0.5)
                        
                        processor = DataPreprocessor('Data')
                        
                        # Step 3: Load datasets from uploaded files
                        status_text.text("📥 Memuat dataset yang diupload...")
                        progress_bar.progress(20)
                        time.sleep(0.5)
                        
                        dataset_loaded = processor.load_datasets()
                        if not dataset_loaded:
                            st.error("❌ Gagal memuat dataset yang diupload. Periksa format file.")
                            st.stop()
                        
                        # Step 4-9: Process data
                        steps = [
                            (30, "🔍 Preprocessing data penjualan..."),
                            (40, "📊 Agregasi bulanan..."),
                            (50, "🔗 Menggabungkan pesanan siap..."),
                            (60, "📋 Menambahkan info produk..."),
                            (70, "📈 Menggabungkan status stok..."),
                            (80, "🏪 Menggabungkan stok toko..."),
                            (90, "🔧 Feature engineering..."),
                            (100, "✅ Finalisasi dataset...")
                        ]
                        
                        for pct, msg in steps:
                            status_text.text(msg)
                            progress_bar.progress(pct)
                            time.sleep(0.3)
                        
                        # Process all data
                        try:
                            success = processor.process_all()
                        except Exception as process_error:
                            st.error(f"❌ Error dalam proses data: {str(process_error)}")
                            success = False
                            
                            # Debug information
                            with st.expander("🔍 Debug Information"):
                                st.write("Error details:")
                                st.code(str(process_error))
                                
                                # Show processor state
                                st.write("Processor state:")
                                for attr in ['hari_libur_df', 'penjualan_df', 'pesanan_siap_df', 'status_stok_df', 'stok_toko_df']:
                                    if hasattr(processor, attr):
                                        df = getattr(processor, attr)
                                        st.write(f"- {attr}: {df.shape if df is not None else 'None'}")
                                    else:
                                        st.write(f"- {attr}: Not found")
                        
                        if success:
                            st.session_state['data_processed'] = True
                            st.session_state['processor'] = processor
                            
                            # Safe way to get dataset info
                            dataset_info = "Dataset berhasil diproses"
                            if hasattr(processor, 'feature_engineering_df') and processor.feature_engineering_df is not None:
                                rows = processor.feature_engineering_df.shape[0]
                                cols = processor.feature_engineering_df.shape[1]
                                dataset_info = f"{rows:,} baris, {cols} kolom"
                            
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>🎉 Data Berhasil Diproses dari File Upload!</h4>
                                <p><strong>Dataset siap:</strong> {dataset_info}</p>
                                <p><strong>File output:</strong> Data/xgboost_ready_dataset.csv</p>
                                <p><strong>File yang diproses:</strong></p>
                                <ul>
                                    <li>📅 {hari_libur_file.name if hari_libur_file else 'N/A'}</li>
                                    <li>🛒 {penjualan_file.name if penjualan_file else 'N/A'}</li>
                                    <li>📦 {pesanan_siap_file.name if pesanan_siap_file else 'N/A'}</li>
                                    <li>📊 {status_stok_file.name if status_stok_file else 'N/A'}</li>
                                    <li>🏪 {stok_toko_file.name if stok_toko_file else 'N/A'}</li>
                                </ul>
                                <p>Anda sekarang dapat melakukan prediksi di menu <strong>📈 Prediksi Penjualan</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.error("❌ Gagal memproses data. Periksa format file yang diupload.")
                            
                    except Exception as e:
                        st.error(f"❌ Error saat memproses data: {str(e)}")
                        with st.expander("🔍 Detail Error"):
                            st.code(str(e))
                
            elif not BACKEND_AVAILABLE:
                st.error("❌ Backend tidak tersedia. Tidak dapat memproses data.")
            else:
                st.warning("⚠️ Mohon upload semua 5 file yang dibutuhkan!")
    
    with tab2:
        st.markdown("### 🔄 Gunakan Data yang Sudah Ada")
        
        if st.button("📊 Muat Dataset Pre-processed", type="secondary"):
            if os.path.exists('Data/xgboost_ready_dataset.csv'):
                with st.spinner("📥 Memuat dataset..."):
                    time.sleep(1)
                    st.session_state['data_processed'] = True
                    
                    # Load basic info
                    df = pd.read_csv('Data/xgboost_ready_dataset.csv')
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Dataset Berhasil Dimuat!</h4>
                        <p><strong>Ukuran dataset:</strong> {df.shape[0]:,} baris × {df.shape[1]} kolom</p>
                        <p><strong>Produk unique:</strong> {df['Produk_Variasi'].nunique():,}</p>
                        <p><strong>Periode data:</strong> {df['Bulan'].min()} - {df['Bulan'].max()}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ File dataset tidak ditemukan. Silakan proses data terlebih dahulu.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "📈 Prediksi Penjualan":
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Prediksi Penjualan Produk</div>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_processed', False):
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Data Belum Dimuat ke Sesi</h4>
            <p>Silakan muat data terlebih dahulu di menu <strong>📁 Upload & Proses Data</strong></p>
            <p>Gunakan tombol <strong>"📊 Muat Dataset Pre-processed"</strong> pada tab <strong>"🔄 Gunakan Data Existing"</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    if not st.session_state.get('model_loaded', False):
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Model Belum Tersedia</h4>
            <p>Model tidak ditemukan. Pastikan file model sudah ada di folder <strong>saved_model/</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Load model dan fitur
    model_path = 'saved_model/xgboost_regressor_TUNED.pkl'
    features_path = 'saved_model/selected_features.pkl'
    data_path = 'Data/xgboost_ready_dataset.csv'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        df = pd.read_csv(data_path)
        df['Bulan'] = pd.to_datetime(df['Bulan'])
        
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ Model dan Data Berhasil Dimuat!</h4>
            <p><strong>Jumlah fitur:</strong> {len(features)}</p>
            <p><strong>Dataset:</strong> {df.shape[0]:,} baris × {df.shape[1]} kolom</p>
            <p><strong>Produk tersedia:</strong> {df['Produk_Variasi'].nunique():,}</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error memuat model/data: {str(e)}")
        st.stop()

    # Interface prediksi
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Konfigurasi Prediksi")
        
        # Pilih produk
        produk_list = sorted(df['Produk_Variasi'].unique())
        produk = st.selectbox(
            "📦 Pilih Produk untuk Diprediksi:", 
            produk_list,
            help="Pilih produk yang ingin diprediksi penjualannya"
        )
    
    with col2:
        st.markdown("### ⚙️ Parameter")
        n_months = st.slider(
            "📅 Jumlah Bulan Prediksi:", 
            min_value=3, 
            max_value=12, 
            value=6,
            help="Jumlah bulan ke depan yang akan diprediksi"
        )

    # Tombol prediksi
    if st.button("🔮 Mulai Prediksi", type="primary", use_container_width=True):
        with st.spinner("🔄 Melakukan prediksi penjualan..."):
            try:
                # Jalankan prediksi
                hasil_prediksi = iterative_forecast_exact(df, model, features, produk, n_months)
                
                if hasil_prediksi:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Prediksi Berhasil!</h4>
                        <p>Prediksi penjualan untuk produk <strong>{produk}</strong> selama {n_months} bulan ke depan telah selesai.</p>
                        <p><em>💡 Catatan: Semua nilai prediksi telah dibulatkan ke bilangan bulat untuk kemudahan interpretasi.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tampilkan hasil prediksi
                    st.markdown("### 📊 Hasil Prediksi")
                    
                    # Create DataFrame for display
                    pred_df = pd.DataFrame(hasil_prediksi)
                    display_df = pred_df[['Bulan', 'Prediksi']].copy()
                    display_df.columns = ['Bulan (YYYY-MM)', 'Prediksi Penjualan']
                    
                    # Show table
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Create historical vs forecast chart
                    st.markdown("### 📈 Visualisasi Historis vs Prediksi")
                    historical_vs_forecast_fig = create_historical_vs_forecast_chart(df, hasil_prediksi, produk)
                    
                    if historical_vs_forecast_fig:
                        st.plotly_chart(historical_vs_forecast_fig, use_container_width=True)
                    
                    # Create separate forecast-only chart
                    st.markdown("### 📊 Detail Prediksi")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pred_df['Bulan'],
                        y=pred_df['Prediksi'],
                        mode='lines+markers',
                        name='Prediksi Penjualan',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8)
                    ))
                    fig.update_layout(
                        title=f'Detail Prediksi Penjualan: {produk}',
                        xaxis_title='Bulan',
                        yaxis_title='Jumlah Penjualan',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    total_pred = sum([r['Prediksi'] for r in hasil_prediksi])
                    avg_pred = total_pred / len(hasil_prediksi)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total Prediksi", f"{int(total_pred):,}")
                    with col2:
                        st.metric("📈 Rata-rata Bulanan", f"{int(avg_pred):,}")
                    with col3:
                        st.metric("📅 Periode Prediksi", f"{n_months} bulan")
                    
                    # Show model information after prediction
                    st.markdown("---")
                    
                    # Algorithm info
                    show_algorithm_info()
                    
                    # Model performance
                    show_model_performance()
                    
                    # Selected features
                    show_selected_features()
                    
                else:
                    st.error("❌ Tidak ada hasil prediksi. Periksa data produk.")
                    
            except Exception as e:
                st.error(f"❌ Error saat prediksi: {str(e)}")
                with st.expander("🔍 Detail Error"):
                    st.code(str(e))
    
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "📊 Analisis Model & Performa":
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Analisis Model & Performa</div>', unsafe_allow_html=True)
    
    if not st.session_state.get('model_loaded', False):
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Model Belum Tersedia</h4>
            <p>Model tidak ditemukan. Pastikan file model sudah ada di folder <strong>saved_model/</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Show comprehensive model analysis
    show_algorithm_info()
    show_model_performance()
    show_selected_features()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p>🏠 <strong>Sistem Prediksi Penjualan Rumah Bayitaz</strong></p>
    <p>Dikembangkan dengan menggunakan Streamlit & XGBoost</p>
    <p>© 2025 - Aplikasi untuk Prediksi Penjualan By InsanPau </p>
</div>
""", unsafe_allow_html=True)
