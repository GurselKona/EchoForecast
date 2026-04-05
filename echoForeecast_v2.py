import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Echo Forecast Pro", layout="wide", page_icon="📈")
st.title("🚀 Echo Forecast Pro - Zaman Makinesi")

# --- SOL MENÜ (UI KONTROLLERİ) ---
st.sidebar.header("⚙️ Analiz Parametreleri")

uploaded_file = st.sidebar.file_uploader("Veri Dosyası Yükle (CSV)", type=["csv"])
target_date_input = st.sidebar.text_input("Hedef Tarih (Örn: 2025-10-15)", value="")
pattern_size = st.sidebar.number_input("Şablon Boyutu (Mum Sayısı)", min_value=10, max_value=200, value=30, step=1)
forward_window = st.sidebar.number_input("Gelecek Penceresi (Mum Sayısı)", min_value=5, max_value=100, value=15, step=1)
threshold_input = st.sidebar.slider("Minimum Uyum Eşiği (%)", min_value=50, max_value=99, value=80, step=1)
threshold = threshold_input / 100.0

run_button = st.sidebar.button("Analizi Çalıştır", type="primary")

# --- ANA ALGORİTMA ---
def run_echo_analysis(file, target_date_str, p_size, f_window, thresh):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    close_col = 'Price' if 'Price' in df.columns else 'Close'
    if df[close_col].dtype == object:
        df[close_col] = df[close_col].astype(str).str.replace(',', '').astype(float)
        
    prices = df[close_col].values
    dates = df['Date'].values

    if target_date_str:
        target_dt = pd.to_datetime(target_date_str)
        past_df = df[df['Date'] <= target_dt]
        if past_df.empty:
            st.error("Girdiğiniz tarih veri setindeki en eski tarihten daha önce.")
            return
        target_idx = past_df.index[-1]
        actual_date_used = df.loc[target_idx, 'Date'].strftime('%Y-%m-%d')
    else:
        target_idx = len(df) - 1
        actual_date_used = df.loc[target_idx, 'Date'].strftime('%Y-%m-%d')

    if target_idx < p_size:
        st.error("Bu tarih için geriye dönük yeterli mum bulunamadı.")
        return

    current_pattern = prices[target_idx - p_size + 1 : target_idx + 1]
    current_dates = dates[target_idx - p_size + 1 : target_idx + 1]
    search_space = prices[: target_idx - p_size + 1]

    actual_future = None
    if target_date_str and (target_idx < len(prices) - 1):
        end_idx = min(target_idx + f_window + 1, len(prices))
        act_prices = prices[target_idx : end_idx]
        actual_future = (act_prices - act_prices[0]) / act_prices[0] * 100

    match_indices = []
    best_r = -1.0
    best_idx = -1
    total_valid_scans = 0 
    
    for i in range(len(search_space) - p_size - f_window + 1):
        past_window = search_space[i : i + p_size]
        if np.std(current_pattern) == 0 or np.std(past_window) == 0:
            continue
        r = np.corrcoef(current_pattern, past_window)[0, 1]
        if np.isnan(r):
            continue
            
        total_valid_scans += 1 
        if r >= thresh:
            match_indices.append(i)
            if r > best_r:
                best_r = r
                best_idx = i

    if best_idx == -1:
        st.warning(f"Belirtilen eşiği (%{thresh*100}) geçen bir tarihsel benzerlik bulunamadı.")
        return

    total_matches = len(match_indices)
    hit_rate = (total_matches / total_valid_scans) * 100 

    entry_idx = best_idx + p_size - 1
    exit_idx = entry_idx + f_window
    entry_price = prices[entry_idx]
    best_trajectory = (prices[entry_idx : exit_idx + 1] - entry_price) / entry_price * 100

    future_trajectories = []
    for idx in match_indices:
        e_idx = idx + p_size - 1
        x_idx = e_idx + f_window
        traj = (prices[e_idx : x_idx + 1] - prices[e_idx]) / prices[e_idx] * 100
        future_trajectories.append(traj)

    current_entry_price = current_pattern[-1]
    best_target_price = current_entry_price * (1 + best_trajectory[-1] / 100)

    future_dates = [pd.to_datetime(current_dates[-1])]
    for i in range(1, f_window + 1):
        if target_idx + i < len(df):
            future_dates.append(pd.to_datetime(df['Date'].iloc[target_idx + i]))
        else:
            next_dt = future_dates[-1] + pd.offsets.BDay(1)
            future_dates.append(next_dt)
            
    future_dates = pd.DatetimeIndex(future_dates)

    # --- METRİKLERİ EKRANA BAS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("En İyi Uyum", f"% {best_r*100:.2f}")
    col2.metric("Hedef Fiyat", f"{best_target_price:.2f}")
    col3.metric("Hedef Tarih", future_dates[-1].strftime('%Y-%m-%d'))
    col4.metric("İsabet Oranı", f"% {hit_rate:.2f}")

    # --- GRAFİĞİ ÇİZ ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(current_dates, current_pattern, label="Güncel Şablon", color='orange', linewidth=3)
    
    for traj in future_trajectories:
        projected_price = current_entry_price * (1 + traj / 100)
        ax.plot(future_dates, projected_price, color='gray', alpha=0.1)
        
    best_projected_price = current_entry_price * (1 + best_trajectory / 100)
    ax.plot(future_dates, best_projected_price, label=f"En İyi Eşleşme", color='blue', linestyle='--', linewidth=3)
    
    if actual_future is not None:
        actual_projected = current_entry_price * (1 + actual_future / 100)
        ax.plot(future_dates[:len(actual_projected)], actual_projected, label="GERÇEKLEŞEN", color='red', linewidth=3.5)

    ax.axvline(x=current_dates[-1], color='black', linestyle=':')
    ax.axhline(y=current_entry_price, color='black', linestyle='--', alpha=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.set_xlim(current_dates[0], future_dates[-1] + (future_dates[-1] - current_dates[0]) * 0.05)
    
    ax.set_title(f"Echo Forecast Analizi (Tarih: {actual_date_used})", fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig) # Streamlit üzerinden grafiği göster

# --- TETİKLEYİCİ ---
if run_button:
    if uploaded_file is not None:
        with st.spinner("Geçmiş taranıyor, lütfen bekleyin..."):
            run_echo_analysis(uploaded_file, target_date_input, pattern_size, forward_window, threshold)
    else:
        st.error("Lütfen önce sol menüden bir CSV dosyası yükleyin.")
else:
    st.info("Sol menüden CSV dosyanızı yükleyin ve parametreleri ayarlayarak 'Analizi Çalıştır' butonuna basın.")