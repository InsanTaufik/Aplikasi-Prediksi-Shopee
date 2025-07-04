# 📊 Sistem Prediksi Penjualan Rumah Bayitaz

## 🎯 Deskripsi Proyek

Aplikasi berbasis web menggunakan Streamlit untuk memprediksi penjualan produk Rumah Bayitaz selama 12 bulan ke depan. Sistem ini menggunakan machine learning XGBoost dengan fitur-fitur canggih seperti prediksi iteratif, analisis musiman, dan lag features yang telah dikembangkan berdasarkan analisis mendalam dari notebook Jupyter.

## 🚀 Fitur Utama

- **🏠 Dashboard Beranda**: Ringkasan sistem dan statistik utama dalam bahasa Indonesia
- **📈 Prediksi Produk Individual**: Prediksi detail 12 bulan untuk produk tertentu
- **📊 Analisis Keseluruhan**: Prediksi multiple produk sekaligus dengan visualisasi agregat
- **🔍 Eksplorasi Data**: Analisis mendalam data historis dan pola penjualan
- **📱 UI Responsif**: Interface bahasa Indonesia yang user-friendly
- **📥 Export Data**: Download hasil prediksi dalam format CSV

## ⚡ Memulai Aplikasi

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
streamlit run Src/app_final_bahasa_indonesia.py
```

Aplikasi akan terbuka di browser di alamat: http://localhost:8501

## 📋 Persyaratan

- Python 3.8+
- Dependencies dalam `requirements.txt`
- Data penjualan historis dalam folder `Data/`
- Model terlatih dalam folder `saved_model/`

## 🏗️ Arsitektur Sistem

```
Data Historis → Feature Engineering → Model XGBoost → Prediksi Iteratif → Visualisasi
     ↓                ↓                 ↓              ↓              ↓
  File CSV      Fitur Temporal     Model Terlatih    12 Bulan     Streamlit UI
               Lag Features (1-3M)  Parameter Terbaik  Forecasting    Chart Interaktif
               Rolling Average      Saved .pkl      Update Dinamis  Download CSV
               Indeks Musiman       Fitur Terpilih   Penyesuaian Musiman   Bahasa Indonesia
```

## 🤖 Model Machine Learning

Sistem menggunakan **XGBoost Regressor** yang telah dioptimasi dengan:

### Fitur-Fitur Utama:
- **Temporal**: Year, Month, Quarter, DaysinMonth, Month_Sin/Cos
- **Lag Features**: Data 1-3 bulan sebelumnya (Total_Jumlah, Penjualan, Views)
- **Rolling Features**: Rata-rata bergulir 3 bulan
- **Seasonal**: Indeks musiman dan tren
- **Product-Specific**: Mean per produk, growth rate, fitur stok

### Metode Prediksi:
- **Iteratif**: Prediksi bulan demi bulan dengan update fitur dinamis
- **Penyesuaian Musiman**: Penyesuaian pola musiman per produk
- **Update Fitur**: Lag dan rolling features diperbarui setiap prediksi

### Evaluasi Model:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error
- **R² Score**: Coefficient of Determination

## 📁 File Input yang Diperlukan
1. `hari_libur_2023-2024.csv` - Data tanggal libur
2. `Penjualan Rumah Bayita Shopee 2023-2024 (FULL).csv` - Data penjualan utama
3. `Pesanan Siap Dikirim 2023-2024.csv` - Data pesanan siap kirim
4. `Status Stok Penjualan 2023-2024 RumahBayitaz.csv` - Data status stok
5. `stok_toko.csv` - Data stok toko

## 🚀 Instruksi Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Buat Model Demo (jika belum memiliki model terlatih)
```bash
cd Src
python create_demo_model.py
```

### 3. Jalankan Aplikasi Streamlit

#### Opsi 1: Menggunakan Script Launcher (Direkomendasikan)
**Windows:**
```cmd
run_app.bat
```

**Python (Semua platform):**
```bash
python run_app.py
```

#### Opsi 2: Perintah Langsung
```bash
streamlit run Src/app_final_bahasa_indonesia.py --server.port 8501
```

## 📌 Fitur Aplikasi

### 🔄 Pemrosesan Data
- **Upload File Otomatis**: Interface upload untuk semua 5 file CSV yang diperlukan
- **Pembersihan Data**: Menangani missing values, parsing tanggal, dan konversi tipe data
- **Feature Engineering**: Membuat 20+ fitur termasuk:
  - Fitur temporal (Year, Month, Quarter, cyclical encoding)
  - Fitur lag (1, 2, 3 bulan)
  - Rolling averages
  - Growth rates
  - Statistik khusus produk
  - Indikator hari libur
  - Rasio harga dan efisiensi

### 🔮 Prediksi
- **Model XGBoost**: Menggunakan XGBoost regressor yang telah dilatih
- **Forecasting Multi-Produk**: Menghasilkan prediksi untuk produk berkinerja terbaik
- **Analisis Pertumbuhan**: Menghitung persentase pertumbuhan
- **Validasi**: Memastikan prediksi tidak negatif

### 📊 Visualisasi
- **Chart Interaktif**: Dibangun dengan Plotly
- **Tampilan Perbandingan**: Penjualan saat ini vs prediksi
- **Analisis Growth Rate**: Indikator visual pertumbuhan
- **Plot Time Series**: Tren historis
- **Desain Responsif**: Bekerja di berbagai ukuran layar

### 💾 Fitur Export
- **Download CSV**: Export prediksi dan data yang diproses
- **File Bermerek Waktu**: Penamaan file otomatis dengan timestamp
- **Multiple Format**: Export ringkasan dan detail

## 🎨 Fitur UI
- **Desain Modern**: Background gradien dan layout kartu
- **Layout Responsif**: Layout lebar dengan navigasi sidebar
- **Indikator Progress**: Loading spinner dan pesan progress
- **Hasil Berkode Warna**: Indikator visual untuk performa
- **Interface Tab**: Konten terorganisir dalam tab

## 🔧 Detail Teknis

### Pipeline Preprocessing Data
1. **Load dan Parse**: Baca file CSV dengan parsing tanggal yang tepat
2. **Agregasi**: Kelompokkan berdasarkan produk dan bulan
3. **Pembuatan Fitur**: Buat fitur time-series dan bisnis
4. **Integrasi Data Eksternal**: Gabungkan dengan data stok dan pengiriman
5. **Penanganan Missing Value**: Isi nilai NaN dengan tepat

### Integrasi Model
- **Loading Joblib**: Serialisasi model yang efisien
- **Alignment Fitur**: Memastikan urutan dan keberadaan fitur yang benar
- **Error Handling**: Penanganan error yang halus untuk fitur yang hilang
- **Prediksi Batch**: Memproses multiple produk secara efisien

### Struktur File
```
├── run_app.bat                       # Windows batch launcher
├── Src/
│   ├── app_final_bahasa_indonesia.py # Aplikasi Streamlit utama
│   ├── core_functions.py             # Fungsi inti backend
│   ├── data_preprocessing.py         # Preprocessing data
│   └── data_comparison.py            # Perbandingan data (opsional)
├── saved_model/
│   ├── xgboost_regressor_TUNED.pkl   # Model XGBoost terlatih
│   └── selected_features.pkl         # Daftar fitur terpilih
├── Data/                             # Upload file CSV Anda di sini
├── requirements.txt                  # Dependencies Python
└── README.md                         # File ini
```

## 🔄 Alur Penggunaan

1. **Mulai Aplikasi**: 
   - **Windows**: Double-click `run_app.bat`
   - **Platform lain**: Jalankan `streamlit run Src/app_final_bahasa_indonesia.py`
2. **Upload File**: Gunakan sidebar untuk upload semua 5 file CSV
3. **Proses Data**: Klik "Proses Data & Generate Prediksi"
4. **Lihat Hasil**: Jelajahi tab yang berbeda:
   - **Overview Data**: Lihat statistik data yang diproses
   - **Prediksi**: Lihat tabel hasil prediksi
   - **Visualisasi**: Chart dan grafik interaktif
   - **Download Hasil**: Export file CSV
5. **Download**: Export hasil untuk analisis lebih lanjut

## 🎯 Fitur Performa
- **Fokus Produk Terbaik**: Menganalisis produk berkinerja terbaik untuk insight yang lebih baik
- **Pemrosesan Efisien**: Dioptimalkan untuk dataset besar
- **Manajemen Memori**: Menangani data secara efisien
- **Prediksi Cepat**: Inferensi model yang cepat

## 🚨 Troubleshooting

### Masalah Umum
1. **Model Tidak Ditemukan**: Pastikan file model ada di folder `saved_model/`
2. **Error Import**: Install semua dependencies dengan `pip install -r requirements.txt`
3. **Masalah Format CSV**: Pastikan file CSV memiliki kolom yang diharapkan
4. **Masalah Memori**: Coba dengan dataset yang lebih kecil terlebih dahulu

### Persyaratan
- Python 3.8+
- RAM 8GB+ direkomendasikan untuk dataset besar
- Browser web modern

## 🔮 Pengembangan Masa Depan
- **Data Real-time**: Koneksi ke sumber data live
- **Model Retraining**: Interface untuk update model
- **Analitik Lanjutan**: Forecasting yang lebih sophisticated
- **Integrasi API**: REST API untuk akses programmatic
- **Dukungan Multi-bahasa**: Internasionalisasi
- **Optimisasi Mobile**: Pengalaman mobile yang lebih baik

## 📝 Lisensi
Proyek ini untuk tujuan edukasi dan penelitian.
