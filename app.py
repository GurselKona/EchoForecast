import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Quant Dashboard", layout="wide", page_icon="🔬")

# Başlık (Daha küçük punto için title yerine header kullanıldı)
st.header("🔬 Quant Dashboard: Echo Forecast & GARCH")

# --- SOL MENÜ (UI KONTROLLERİ) ---
st.sidebar.header("⚙️ Parametreler")

uploaded_file = st.sidebar.file_uploader("Veri Dosyası Yükle (CSV)", type=["csv"])

df = None

# Session State Yönetimi (Otomatik Tarih İçin)
if "target_date_val" not in st.session_state:
    st.session_state.target_date_val = ""

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    if "loaded_filename" not in st.session_state or st.session_state.loaded_filename != uploaded_file.name:
        st.session_state.target_date_val = df['Date'].iloc[-1].strftime('%Y-%m-%d')
        st.session_state.loaded_filename = uploaded_file.name

target_date_input = st.sidebar.text_input("Hedef Tarih", key="target_date_val")

# YENİ: Boyut parametrelerini dikeyden tasarruf için YAN YANA (columns) alıyoruz
col1, col2 = st.sidebar.columns(2)
pattern_size = col1.number_input("Şablon", min_value=10, max_value=200, value=30, step=1)
forward_window = col2.number_input("Gelecek", min_value=5, max_value=100, value=15, step=1)

threshold_input = st.sidebar.slider("Minimum Uyum Eşiği (%)", min_value=50, max_value=99, value=80, step=1)
threshold = threshold_input / 100.0

st.sidebar.markdown("---")
st.sidebar.caption("GARCH(1,1) Ayarları")

# YENİ: GARCH parametrelerini YAN YANA alıyoruz
col3, col4 = st.sidebar.columns(2)
garch_p = col3.number_input("P (Varyans)", min_value=1, max_value=5, value=1)
garch_q = col4.number_input("Q (Hata)", min_value=1, max_value=5, value=1)

st.sidebar.markdown("<br>", unsafe_allow_html=True) # Butonun üstüne ufak bir boşluk
run_button = st.sidebar.button("Her İki Modeli Çalıştır", type="primary", use_container_width=True)

# --- ANA ALGORİTMA ---
def run_quant_analysis(df_input, target_date_str, p_size, f_window, thresh, g_p, g_q):
    df_main = df_input.copy()
    
    close_col = 'Price' if 'Price' in df_main.columns else 'Close'
    if df_main[close_col].dtype == object:
        df_main[close_col] = df_main[close_col].astype(str).str.replace(',', '').astype(float)
        
    prices = df_main[close_col].values
    dates = df_main['Date'].values

    if target_date_str:
        target_dt = pd.to_datetime(target_date_str)
        past_df = df_main[df_main['Date'] <= target_dt]
        if past_df.empty:
            st.error("HATA: Tarih veri setinden daha eski.")
            return
        target_idx = past_df.index[-1]
        actual_date_used = df_main.loc[target_idx, 'Date'].strftime('%Y-%m-%d')
    else:
        target_idx = len(df_main) - 1
        actual_date_used = df_main.loc[target_idx, 'Date'].strftime('%Y-%m-%d')

    if target_idx < p_size:
        st.error("HATA: Yeterli geçmiş mum yok.")
        return

    current_dates = dates[target_idx - p_size + 1 : target_idx + 1]
    future_dates = [pd.to_datetime(current_dates[-1])]
    for i in range(1, f_window + 1):
        if target_idx + i < len(df_main):
            future_dates.append(pd.to_datetime(df_main['Date'].iloc[target_idx + i]))
        else:
            future_dates.append(future_dates[-1] + pd.offsets.BDay(1))
    future_dates = pd.DatetimeIndex(future_dates)
    
    current_entry_price = prices[target_idx]

    actual_projected = None
    if target_idx < len(prices) - 1:
        end_idx = min(target_idx + f_window + 1, len(prices))
        actual_projected = prices[target_idx : end_idx]

    tab1, tab2 = st.tabs(["📊 ECHO FORECAST", "🌪️ GARCH VOLATİLİTE"])

    # =======================================================
    # TAB 1: ECHO FORECAST
    # =======================================================
    with tab1:
        current_pattern = prices[target_idx - p_size + 1 : target_idx + 1]
        search_space = prices[: target_idx - p_size + 1]

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
            st.warning(f"Belirtilen eşiği (%{thresh*100}) geçen bir benzerlik bulunamadı.")
        else:
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

            best_target_price = current_entry_price * (1 + best_trajectory[-1] / 100)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("En İyi Uyum", f"% {best_r*100:.2f}")
            c2.metric("Hedef Fiyat", f"{best_target_price:.2f}")
            c3.metric("Taranan / Bulunan", f"{total_valid_scans} / {total_matches}")
            c4.metric("İsabet Oranı", f"% {hit_rate:.2f}")

            # Dikeyde daha dar grafik (figsize=(12, 4))
            fig_echo, ax_echo = plt.subplots(figsize=(12, 4))
            ax_echo.plot(current_dates, current_pattern, label="Güncel Şablon", color='orange', linewidth=2.5)
            
            for traj in future_trajectories:
                ax_echo.plot(future_dates, current_entry_price * (1 + traj / 100), color='gray', alpha=0.1)
                
            best_projected_price = current_entry_price * (1 + best_trajectory / 100)
            ax_echo.plot(future_dates, best_projected_price, label="En İyi Eşleşme", color='blue', linestyle='--', linewidth=2.5)
            
            if actual_projected is not None:
                ax_echo.plot(future_dates[:len(actual_projected)], actual_projected, label="GERÇEKLEŞEN", color='red', linewidth=3)

            ax_echo.axvline(x=current_dates[-1], color='black', linestyle=':')
            ax_echo.axhline(y=current_entry_price, color='black', linestyle='--', alpha=0.5)
            ax_echo.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_echo.legend(loc='lower left', prop={'size': 9})
            ax_echo.grid(True, alpha=0.3)
            fig_echo.autofmt_xdate(rotation=45)
            st.pyplot(fig_echo)

    # =======================================================
    # TAB 2: GARCH (1,1)
    # =======================================================
    with tab2:
        try:
            hist_prices = prices[:target_idx+1]
            returns = np.diff(hist_prices) / hist_prices[:-1] * 100 
            
            am = arch_model(returns, vol='Garch', p=g_p, q=g_q, dist='normal')
            res = am.fit(disp='off', update_freq=0)
            
            forecasts = res.forecast(horizon=f_window, reindex=False)
            
            f_mean = forecasts.mean.iloc[-1].values
            f_var = forecasts.variance.iloc[-1].values
            f_std = np.sqrt(f_var) 

            garch_mean_path = current_entry_price * np.cumprod(1 + (f_mean / 100))
            garch_upper_path = current_entry_price * np.cumprod(1 + ((f_mean + 1.96 * f_std) / 100))
            garch_lower_path = current_entry_price * np.cumprod(1 + ((f_mean - 1.96 * f_std) / 100))

            garch_mean_path = np.insert(garch_mean_path, 0, current_entry_price)
            garch_upper_path = np.insert(garch_upper_path, 0, current_entry_price)
            garch_lower_path = np.insert(garch_lower_path, 0, current_entry_price)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Güncel Oynaklık", f"% {np.std(returns[-30:]):.2f}")
            c2.metric("Beklenen Ortalama", f"{garch_mean_path[-1]:.2f}")
            c3.metric("Maksimum Risk (Tavan)", f"{garch_upper_path[-1]:.2f}")
            c4.metric("Maksimum Risk (Taban)", f"{garch_lower_path[-1]:.2f}")

            fig_garch, ax_garch = plt.subplots(figsize=(12, 4))
            
            ax_garch.plot(current_dates, current_pattern, label="Geçmiş Fiyat", color='black', linewidth=2)
            ax_garch.fill_between(future_dates, garch_lower_path, garch_upper_path, color='green', alpha=0.2, label="%95 Güven Aralığı")
            ax_garch.plot(future_dates, garch_mean_path, color='green', linestyle='--', linewidth=2, label="Beklenen Ortalama")
            
            if actual_projected is not None:
                ax_garch.plot(future_dates[:len(actual_projected)], actual_projected, label="GERÇEKLEŞEN", color='red', linewidth=3)

            ax_garch.axvline(x=current_dates[-1], color='black', linestyle=':')
            ax_garch.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_garch.legend(loc='upper left', prop={'size': 9})
            ax_garch.grid(True, alpha=0.3)
            fig_garch.autofmt_xdate(rotation=45)
            st.pyplot(fig_garch)

        except Exception as e:
            st.error(f"GARCH Modeli çalıştırılırken bir hata oluştu: {e}")

# --- TETİKLEYİCİ ---
if run_button:
    if df is not None:
        with st.spinner("Hesaplanıyor..."):
            run_quant_analysis(df, st.session_state.target_date_val, pattern_size, forward_window, threshold, garch_p, garch_q)
    else:
        st.error("Lütfen önce sol menüden bir CSV dosyası yükleyin.")
else:
    st.info("Sol menüden CSV dosyanızı yükleyin ve 'Her İki Modeli Çalıştır' butonuna basın.")