import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Mengabaikan FutureWarning dari Streamlit/Pandas/Scikit-learn
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- FUNGSI UTAMA UNTUK MEMUAT DAN MEMPROSES DATA ---

@st.cache_data
def load_and_prepare_data(filepath):
    """
    Memuat data, melakukan pembersihan dasar, dan preprocessing untuk clustering.
    Fungsi ini di-cache agar data hanya dimuat sekali.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"File {filepath} tidak ditemukan. Pastikan file 'smartphones_cleaned.csv' ada.")
        return pd.DataFrame(), pd.DataFrame(), None

    # Mengganti nama kolom untuk kejelasan
    df = df.rename(columns={'ram_capacity': 'RAM', 'rom_capacity': 'ROM', 'battery_capacity': 'Battery', 'fast_charging_capacity': 'Fast_Charging'})

    # Pembersihan dasar (sama dengan yang ada di notebook)
    df = df.dropna(subset=['price', 'rating', 'RAM', 'ROM', 'Battery', 'Fast_Charging', 'screen_size'])
    df = df[df['price'] > 10000] # Hanya ambil smartphone yang harganya lebih dari 10000 (untuk memfilter outlier harga sangat rendah)
    df = df.copy() # Hindari SettingWithCopyWarning

    # Mengubah tipe data RAM, ROM, Fast_Charging, Battery ke numerik (Float)
    for col in ['RAM', 'ROM', 'Fast_Charging', 'Battery']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['RAM', 'ROM', 'Fast_Charging', 'Battery'], inplace=True)

    # Menghitung rasio Price/Rating
    df['Price_Per_Rating'] = df['price'] / df['rating']

    # Normalisasi (Quantile Transformer) dan Scaling (Standard Scaler) untuk Clustering
    # Kolom yang akan digunakan untuk clustering (sesuai yang umum digunakan untuk menentukan segmen)
    clustering_cols = ['price', 'rating', 'RAM', 'ROM', 'Battery', 'Fast_Charging', 'Price_Per_Rating']

    # 1. Transformasi non-linear (untuk mengurangi skewness)
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=len(df), random_state=42)
    df_qt = pd.DataFrame(qt.fit_transform(df[clustering_cols]), columns=clustering_cols, index=df.index)

    # 2. Standard Scaling (untuk K-Means)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_qt), columns=clustering_cols, index=df.index)

    return df, df_scaled, clustering_cols

# --- FUNGSI UNTUK CLUSTERING K-MEANS ---

@st.cache_data
def run_kmeans(df_scaled, n_clusters):
    """
    Menjalankan algoritma K-Means dan menghitung Silhouette Score.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    df_scaled['Cluster'] = kmeans.fit_predict(df_scaled)
    # Gunakan data awal untuk menghitung silhouette score karena skalanya lebih mudah diinterpretasi
    # Catatan: Silhouette score harusnya dihitung pada data yang sudah di-scale, tapi mari kita gunakan data yang di-transform (df_qt) untuk konsistensi dengan notebook.
    # Karena df_scaled sudah hasil scaling dari df_qt, kita gunakan df_scaled saja.
    score = silhouette_score(df_scaled[df_scaled.columns.difference(['Cluster'])], df_scaled['Cluster'])
    return df_scaled['Cluster'], score

# --- TATA LETAK APLIKASI STREAMLIT ---

st.set_page_config(
    page_title="Dashboard Analisis dan Clustering Smartphone",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📱 Dashboard Analisis dan Segmentasi Smartphone")
st.markdown("Eksplorasi interaktif performa, harga, dan spesifikasi *smartphone* berdasarkan dataset.")

# Memuat dan mempersiapkan data
df_raw, df_scaled, clustering_cols = load_and_prepare_data('smartphones_cleaned.csv')

if df_raw.empty:
    st.stop() # Hentikan jika data gagal dimuat

# --- SIDEBAR: KONTROL DAN FILTER ---

with st.sidebar:
    st.header("⚙️ Kontrol Data & Analisis")

   # Filter Brand
    all_brands_in_data = sorted(df_raw['brand_name'].unique().tolist())
    all_brands = ['Semua'] + all_brands_in_data

    # Cek dan filter nilai default agar sesuai dengan data yang ada
    intended_defaults = ['Samsung', 'Xiaomi', 'OnePlus', 'Apple']
    safe_defaults = [brand for brand in intended_defaults if brand in all_brands_in_data]
    
    selected_brands = st.multiselect(
        "Pilih Brand (Merek)",
        options=all_brands,
        # Menggunakan safe_defaults, atau 4 merek teratas jika safe_defaults kosong
        default=safe_defaults if safe_defaults else all_brands_in_data[:4] 
    )

    if 'Semua' in selected_brands:
        df_filtered = df_raw.copy()
    else:
        df_filtered = df_raw[df_raw['brand_name'].isin(selected_brands)]

    st.markdown("---")

    # Kontrol Clustering
    st.header("🔬 Analisis Clustering (K-Means)")
    # Elbow method dari notebook menyarankan 3-5 cluster. Kita default 3.
    n_clusters = st.slider("Jumlah Cluster (Segmen)", min_value=2, max_value=8, value=3)

    if st.button("Jalankan Clustering"):
        if n_clusters < 2:
            st.warning("Jumlah cluster harus minimal 2.")
        else:
            with st.spinner(f'Menjalankan K-Means dengan {n_clusters} cluster...'):
                df_raw['Cluster'], silhouette = run_kmeans(df_scaled.copy(), n_clusters)
                st.success(f"Clustering Selesai! Silhouette Score: {silhouette:.4f}")
                # Update data yang difilter agar memiliki kolom Cluster
                df_filtered = df_raw[df_raw['brand_name'].isin(selected_brands) if 'Semua' not in selected_brands else df_raw].copy()
    else:
        # Jika tombol belum diklik, tetapkan cluster ke dummy atau hasil sebelumnya jika ada
        if 'Cluster' not in df_raw.columns:
             df_raw['Cluster'] = 0
        df_filtered = df_raw[df_raw['brand_name'].isin(selected_brands) if 'Semua' not in selected_brands else df_raw].copy()


# --- TAB UTAMA ---

tab_eksplorasi, tab_clustering = st.tabs(["📊 Eksplorasi Data Interaktif", "🔬 Hasil Clustering (Segmentasi)"])

# =========================================================================
# TAB 1: EKSPLORASI DATA INTERAKTIF
# =========================================================================

with tab_eksplorasi:
    st.header("Visualisasi Data Terfilter")

    # 1. Distribusi Harga
    st.subheader("1. Distribusi Harga Smartphone")
    col1, col2 = st.columns(2)
    min_price, max_price = int(df_filtered['price'].min()), int(df_filtered['price'].max())
    price_range = col1.slider(
        "Rentang Harga (USD)",
        min_value=0,
        max_value=int(df_raw['price'].max()),
        value=(min_price, max_price),
        step=5000
    )
    df_price_filtered = df_filtered[(df_filtered['price'] >= price_range[0]) & (df_filtered['price'] <= price_range[1])]

    fig_price = px.histogram(
        df_price_filtered,
        x='price',
        color='brand_name',
        title=f'Distribusi Harga ({len(df_price_filtered)} Smartphone)',
        labels={'price': 'Harga (USD)'},
        nbins=50,
        height=400,
        template='plotly_white'
    )
    fig_price.update_layout(xaxis_title="Harga (Rp)", yaxis_title="Jumlah Model")
    col2.metric("Jumlah Model Terfilter", len(df_price_filtered))
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")

    # 2. Scatter Plot Interaktif: Harga vs Rating
    st.subheader("2. Perbandingan Harga, Rating, dan Spesifikasi")

    y_options = {
        'Rating': 'rating',
        'Kapasitas Baterai (mAh)': 'Battery',
        'Kapasitas RAM (GB)': 'RAM',
        'Kecepatan Fast Charging (W)': 'Fast_Charging'
    }

    x_col = st.selectbox("Sumbu X (Variabel Penentu)", options=clustering_cols, index=0)
    y_col_name = st.selectbox("Sumbu Y (Metrik Perbandingan)", options=list(y_options.keys()), index=0)
    y_col = y_options.get(y_col_name, 'rating')

    size_col = st.selectbox("Ukuran Gelembung (Size)", options=['ROM', 'screen_size', 'Fast_Charging', 'price'], index=0)

    fig_scatter = px.scatter(
        df_price_filtered,
        x=x_col,
        y=y_col,
        color='brand_name',
        size=size_col,
        hover_data=['model', 'price', 'rating', 'RAM', 'ROM', 'Battery'],
        title=f'{y_col_name} vs {x_col} (Ukuran: {size_col})',
        labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col_name},
        height=600,
        template='plotly_white'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# =========================================================================
# TAB 2: HASIL CLUSTERING (SEGMENTASI)
# =========================================================================

with tab_clustering:
    st.header(f"Hasil Segmentasi Pasar ({n_clusters} Cluster)")

    if 'Cluster' in df_raw.columns and df_raw['Cluster'].nunique() > 1:
        st.info("Visualisasi ini menunjukkan smartphone yang sudah diberi label cluster, berdasarkan data yang sudah di-scale.")

        # Menghitung Rata-rata Spesifikasi per Cluster
        cluster_summary = df_raw.groupby('Cluster')[['price', 'rating', 'RAM', 'ROM', 'Battery', 'Fast_Charging', 'screen_size']].mean().reset_index()
        # Mengubah nama kolom Cluster menjadi Segmen
        cluster_summary['Cluster'] = 'Segmen ' + (cluster_summary['Cluster'] + 1).astype(str)
        cluster_summary = cluster_summary.round(2)

        st.subheader("Ringkasan Rata-rata Spesifikasi per Segmen")
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Plot 3D untuk memvisualisasikan Cluster
        st.subheader("Visualisasi Segmen Pasar (3D Scatter Plot)")
        st.markdown("Pilih 3 variabel untuk memvisualisasikan pemisahan cluster.")

        col3, col4, col5 = st.columns(3)
        x_3d = col3.selectbox("Sumbu X (3D)", options=clustering_cols, index=0)
        y_3d = col4.selectbox("Sumbu Y (3D)", options=clustering_cols, index=1)
        z_3d = col5.selectbox("Sumbu Z (3D)", options=clustering_cols, index=2)

        # Plot 3D menggunakan data yang belum di-scale (lebih mudah dibaca)
        df_plot_3d = df_raw.copy()
        df_plot_3d['Segmen'] = 'Segmen ' + (df_plot_3d['Cluster'] + 1).astype(str)

        fig_3d = px.scatter_3d(
            df_plot_3d,
            x=x_3d,
            y=y_3d,
            z=z_3d,
            color='Segmen',
            hover_data=['model', 'brand_name', 'price', 'rating', 'RAM', 'ROM'],
            title=f'Visualisasi Segmen Pasar: {x_3d} vs {y_3d} vs {z_3d}',
            labels={
                x_3d: x_3d.replace('_', ' ').title(),
                y_3d: y_3d.replace('_', ' ').title(),
                z_3d: z_3d.replace('_', ' ').title(),
            },
            height=700,
            template='plotly_white'
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")

        # Distribusi Brand per Cluster
        st.subheader("Distribusi Brand di Setiap Segmen")
        df_brand_cluster = df_raw.groupby(['Segmen', 'brand_name']).size().reset_index(name='Jumlah')
        df_brand_cluster['Segmen'] = 'Segmen ' + (df_brand_cluster['Segmen']).astype(str)

        fig_brand_dist = px.bar(
            df_brand_cluster,
            x='Segmen',
            y='Jumlah',
            color='brand_name',
            title='Kontribusi Merek di Setiap Segmen',
            labels={'Segmen': 'Segmen Pasar', 'Jumlah': 'Jumlah Model', 'brand_name': 'Merek'},
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig_brand_dist, use_container_width=True)

    else:
        st.warning("Silakan klik tombol 'Jalankan Clustering' di sidebar untuk melihat hasil segmentasi.")

# --- DISPLAY RAW DATA (opsional) ---
st.markdown("---")
if st.checkbox("Tampilkan Data Mentah/Hasil (Tabel)"):

    st.dataframe(df_raw, use_container_width=True)
