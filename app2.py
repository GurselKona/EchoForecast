import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Quant Dashboard", layout="wide", page_icon="🔬")
st.header("🔬 Quant Dashboard: Echo, Volatilite & TimesFM")

# --- YAPAY ZEKA (TIMESFM) ÖNBELLEK YÖNETİMİ ---
@st.cache_resource(show_spinner=False)
def load_timesfm_model(horizon_len=15):
    try:
        import timesfm
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=1, 
                horizon_len=horizon_len,
                context_len=512, 
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
            )
        )
        return tfm, True
    except Exception as e:
        import traceback
        return traceback.format_exc(), False

# --- SOL MENÜ (UI KONTROLLERİ) ---
st.sidebar.header("⚙️ Parametreler")
uploaded_file = st.sidebar.file_uploader("Veri Dosyası Yükle (CSV)", type=["csv"])
df = None

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

col1, col2 = st.sidebar.columns(2)
pattern_size = col1.number_input("Şablon", min_value=10, max_value=200, value=30, step=1)
forward_window = col2.number_input("Gelecek", min_value=5, max_value=100, value=15, step=1)
threshold_input = st.sidebar.slider("Minimum Uyum Eşiği (%)", min_value=50, max_value=99, value=80, step=1)
threshold = threshold_input / 100.0

st.sidebar.markdown("---")
st.sidebar.caption("TimesFM (Yapay Zeka) Ayarları")
# Kullanıcıya 3 farklı güven aralığı sunuyoruz
tfm_band_choice = st.sidebar.selectbox(
    "AI Güven Aralığı (Olasılık Bandı)", 
    ["Dar (%30 - %70)", "Orta (%20 - %80)", "Geniş (%10 - %90)"], 
    index=2 # Varsayılan olarak Geniş gelsin
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
run_button = st.sidebar.button("Modellerin Tamamını Çalıştır", type="primary", use_container_width=True)

st.sidebar.caption("Volatilite (Risk) Ayarları")

# Risk Modeli Seçimi (vol_model_type burada tanımlanır)
vol_model_type = st.sidebar.selectbox("Risk Modeli", ["GARCH", "EGARCH"])

# GARCH P ve Q değerleri (garch_q burada tanımlanır)
col3, col4 = st.sidebar.columns(2)
garch_p = col3.number_input("P (Varyans)", min_value=1, max_value=5, value=1)
garch_q = col4.number_input("Q (Hata)", min_value=1, max_value=5, value=1)

# --- GRAFİK AYARLARI FONKSİYONU (SİMSİYAH EKSENLER) ---
def apply_plotly_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(color='black', size=18, family="Arial Black")),
        height=500,
        # YENİ: l(sol), r(sağ), b(alt) boşlukları 20'den 80'e çıkarıldı ki yazılar nefes alsın
        margin=dict(l=80, r=80, t=50, b=80), 
        hovermode="x unified", 
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='#E5E5E5',
        showline=True, linewidth=2, linecolor='black', 
        showspikes=True, spikemode="across", spikedash="dot", spikecolor="black", spikethickness=1,
        tickfont=dict(color='black', size=15, family="Arial Black"),
        automargin=True # YENİ: X ekseni yazıları sığmazsa ekranı otomatik yukarı iter
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='#E5E5E5',
        showline=True, linewidth=2, linecolor='black', 
        showspikes=True, spikemode="across", spikedash="dot", spikecolor="black", spikethickness=1,
        tickfont=dict(color='black', size=15, family="Arial Black"),
        automargin=True # YENİ: Y ekseni yazıları sığmazsa grafiği otomatik sağa iter
    )
    return fig

# --- ANA ALGORİTMA ---
def run_quant_analysis(df_input, target_date_str, p_size, f_window, thresh, g_p, g_q, vol_type, tfm_band_choice):
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
    else:
        target_idx = len(df_main) - 1

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

    tab1, tab2, tab3 = st.tabs(["📊 ECHO FORECAST", f"🌪️ {vol_type} VOLATİLİTE", "🧠 TIMESFM (Derin Öğrenme)"])

    # =======================================================
    # TAB 1: ECHO FORECAST (PLOTLY)
    # =======================================================
    with tab1:
        current_pattern = prices[target_idx - p_size + 1 : target_idx + 1]
        search_space = prices[: target_idx - p_size + 1]
        match_indices = []
        best_r, best_idx, total_valid_scans = -1.0, -1, 0
        
        for i in range(len(search_space) - p_size - f_window + 1):
            past_window = search_space[i : i + p_size]
            if np.std(current_pattern) == 0 or np.std(past_window) == 0: continue
            r = np.corrcoef(current_pattern, past_window)[0, 1]
            if np.isnan(r): continue
            total_valid_scans += 1 
            if r >= thresh:
                match_indices.append(i)
                if r > best_r: best_r, best_idx = r, i

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

            fig_echo = go.Figure()
            fig_echo.add_trace(go.Scatter(x=current_dates, y=current_pattern, mode='lines', name='Güncel Şablon', line=dict(color='orange', width=3)))
            for i, traj in enumerate(future_trajectories):
                show_leg = True if i == 0 else False
                fig_echo.add_trace(go.Scatter(x=future_dates, y=current_entry_price * (1 + traj / 100), mode='lines', name='Alternatif Rotalar', line=dict(color='gray', width=1), opacity=0.2, showlegend=show_leg, hoverinfo='skip'))
            best_projected_price = current_entry_price * (1 + best_trajectory / 100)
            fig_echo.add_trace(go.Scatter(x=future_dates, y=best_projected_price, mode='lines', name='En İyi Eşleşme', line=dict(color='blue', width=3, dash='dash')))
            if actual_projected is not None:
                fig_echo.add_trace(go.Scatter(x=future_dates[:len(actual_projected)], y=actual_projected, mode='lines', name='GERÇEKLEŞEN', line=dict(color='red', width=3)))
            
            fig_echo = apply_plotly_layout(fig_echo, "Echo Forecast Projeksiyonu")
            # YENİ: theme=None ile Streamlit'in ayarlarımızı ezmesi yasaklandı!
            st.plotly_chart(fig_echo, use_container_width=True, theme=None)

    # =======================================================
    # TAB 2: VOLATİLİTE (PLOTLY)
    # =======================================================
    with tab2:
        try:
            hist_prices = prices[:target_idx+1]
            returns = np.diff(hist_prices) / hist_prices[:-1] * 100 
            
            if vol_type == "EGARCH":
                am = arch_model(returns, vol='EGARCH', p=g_p, o=1, q=g_q, dist='normal')
                res = am.fit(disp='off', update_freq=0)
                forecasts = res.forecast(horizon=f_window, reindex=False, method='simulation')
            else:
                am = arch_model(returns, vol='GARCH', p=g_p, q=g_q, dist='normal')
                res = am.fit(disp='off', update_freq=0)
                forecasts = res.forecast(horizon=f_window, reindex=False, method='analytic')
            
            f_mean = forecasts.mean.iloc[-1].values
            f_std = np.sqrt(forecasts.variance.iloc[-1].values) 

            garch_mean_path = current_entry_price * np.cumprod(1 + (f_mean / 100))
            garch_upper_path = current_entry_price * np.cumprod(1 + ((f_mean + 1.96 * f_std) / 100))
            garch_lower_path = current_entry_price * np.cumprod(1 + ((f_mean - 1.96 * f_std) / 100))

            garch_mean_path = np.insert(garch_mean_path, 0, current_entry_price)
            garch_upper_path = np.insert(garch_upper_path, 0, current_entry_price)
            garch_lower_path = np.insert(garch_lower_path, 0, current_entry_price)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Güncel Oynaklık", f"% {np.std(returns[-30:]):.2f}")
            c2.metric("Beklenen Ortalama", f"{garch_mean_path[-1]:.2f}")
            c3.metric("Maks. Risk (Tavan)", f"{garch_upper_path[-1]:.2f}")
            c4.metric("Maks. Risk (Taban)", f"{garch_lower_path[-1]:.2f}")

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=current_dates, y=current_pattern, mode='lines', name='Geçmiş Fiyat', line=dict(color='black', width=2)))
            
            fig_vol.add_trace(go.Scatter(x=future_dates, y=garch_upper_path, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_vol.add_trace(go.Scatter(x=future_dates, y=garch_lower_path, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 128, 0, 0.2)', name=f'%95 Güven Aralığı ({vol_type})'))
            
            fig_vol.add_trace(go.Scatter(x=future_dates, y=garch_mean_path, mode='lines', name='Beklenen Ortalama', line=dict(color='green', width=2, dash='dash')))
            if actual_projected is not None:
                fig_vol.add_trace(go.Scatter(x=future_dates[:len(actual_projected)], y=actual_projected, mode='lines', name='GERÇEKLEŞEN', line=dict(color='red', width=3)))

            fig_vol = apply_plotly_layout(fig_vol, f"{vol_type} Risk Konisi")
            # YENİ: theme=None
            st.plotly_chart(fig_vol, use_container_width=True, theme=None)

        except Exception as e:
            st.error(f"{vol_type} Modeli çalıştırılamadı: {e}")

   # =======================================================
    # TAB 3: TIMESFM (PLOTLY - GETİRİ DÖNÜŞÜMLÜ / STATIONARY)
    # =======================================================
    with tab3:
        # 1. YAPAY ZEKA İÇİN VERİ DÖNÜŞÜMÜ (Ham Fiyat -> Yüzdesel Getiri)
        hist_prices = prices[:target_idx+1]
        tfm_returns = np.diff(hist_prices) / hist_prices[:-1] * 100 
        
        tfm_context_len = min(512, len(tfm_returns)) 
        
        if tfm_context_len < 32:
            st.warning("TimesFM'in çalışabilmesi için hedeften önce en az 33 mum geçmiş veri olmalıdır.")
        else:
            with st.spinner("TimesFM Derin Öğrenme Modeli Başlatılıyor (Durağan Veri ile)..."):
                tfm_model, is_loaded = load_timesfm_model(horizon_len=f_window)
                if not is_loaded:
                    st.error(f"TimesFM yüklenemedi: {tfm_model}")
                else:
                    try:
                        # Modele sadece getirileri veriyoruz
                        tfm_input_data = tfm_returns[-tfm_context_len:]
                        
                        forecast_result = tfm_model.forecast([tfm_input_data], freq=[0])
                        
                        # Çıkan sonuçlar artık FİYAT DEĞİL, BEKLENEN GETİRİ ORANLARIDIR
                        point_forecast_returns = forecast_result[0][0]
                        quantiles_returns = forecast_result[1][0] 
                        
                        if tfm_band_choice == "Geniş (%10 - %90)":
                            q_lower_idx, q_upper_idx = 0, 8
                            band_label = "%10 - %90"
                        elif tfm_band_choice == "Orta (%20 - %80)":
                            q_lower_idx, q_upper_idx = 1, 7
                            band_label = "%20 - %80"
                        else:
                            q_lower_idx, q_upper_idx = 2, 6
                            band_label = "%30 - %70"
                        
                        # 2. MATEMATİKSEL İNŞA (Getiriyi Tekrar Fiyata Çevirme)
                        # Kümülatif çarpım (Compound) ile fiyata giydiriyoruz
                        tfm_mean_path = current_entry_price * np.cumprod(1 + (point_forecast_returns / 100))
                        tfm_upper_path = current_entry_price * np.cumprod(1 + (quantiles_returns[:, q_upper_idx] / 100))
                        tfm_lower_path = current_entry_price * np.cumprod(1 + (quantiles_returns[:, q_lower_idx] / 100))
                        
                        # Grafikte kopukluk olmasın diye 0. güne (bugün) giriş fiyatını ekliyoruz
                        tfm_mean_path = np.insert(tfm_mean_path, 0, current_entry_price)
                        tfm_upper_path = np.insert(tfm_upper_path, 0, current_entry_price) 
                        tfm_lower_path = np.insert(tfm_lower_path, 0, current_entry_price) 
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Giriş Fiyatı", f"{current_entry_price:.2f}")
                        c2.metric("Yapay Zeka Ort. Hedef", f"{tfm_mean_path[-1]:.2f}")
                        c3.metric(f"AI {band_label.split(' - ')[1]} İyimser Senaryo", f"{tfm_upper_path[-1]:.2f}")
                        c4.metric(f"AI {band_label.split(' - ')[0]} Kötümser Senaryo", f"{tfm_lower_path[-1]:.2f}")

                        fig_tfm = go.Figure()
                        plot_past_dates = current_dates[-p_size:]
                        plot_past_prices = current_pattern[-p_size:]
                        
                        fig_tfm.add_trace(go.Scatter(x=plot_past_dates, y=plot_past_prices, mode='lines', name='Geçmiş Fiyat', line=dict(color='black', width=2)))
                        
                        fig_tfm.add_trace(go.Scatter(x=future_dates, y=tfm_upper_path, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig_tfm.add_trace(go.Scatter(x=future_dates, y=tfm_lower_path, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31, 119, 180, 0.2)', name=f'AI Olasılık Bandı ({band_label})'))
                        
                        fig_tfm.add_trace(go.Scatter(x=future_dates, y=tfm_mean_path, mode='lines', name='TimesFM Nokta Tahmini', line=dict(color='#1f77b4', width=2.5, dash='dash')))
                        
                        if actual_projected is not None:
                            fig_tfm.add_trace(go.Scatter(x=future_dates[:len(actual_projected)], y=actual_projected, mode='lines', name='GERÇEKLEŞEN', line=dict(color='red', width=3)))

                        fig_tfm = apply_plotly_layout(fig_tfm, "TimesFM (Durağan Getiri Modeli)")
                        st.plotly_chart(fig_tfm, use_container_width=True, theme=None)

                    except Exception as e:
                        st.error(f"TimesFM tahmini oluşturulurken hata: {e}")

if run_button:
    if df is not None:
        with st.spinner("Modeller Hesaplanıyor..."):
            # Sıralama Eşleşmesi:
            # garch_p        --> içeriye g_p olarak girer
            # garch_q        --> içeriye g_q olarak girer
            # vol_model_type --> içeriye vol_type olarak girer
            # tfm_band_choice -> içeriye aynen girer
            run_quant_analysis(df, st.session_state.target_date_val, pattern_size, forward_window, threshold, garch_p, garch_q, vol_model_type, tfm_band_choice)
    else:
        st.error("Lütfen önce sol menüden bir CSV dosyası yükleyin.")