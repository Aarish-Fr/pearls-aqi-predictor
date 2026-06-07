# =============================================================================
# Pearls AQI Predictor — dashboard.py
# =============================================================================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["OMP_NUM_THREADS"]       = "1"

import streamlit as st
import time

# ── Page config — must be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Eager UI Render Hack for Slow Imports ────────────────────────────────────
boot_box = st.empty()
if "imports_done" not in st.session_state:
    boot_box.info("🚀 **System Booting:** Initializing Visualization & Backend engines. This may take ~30 seconds on cold start. Please wait...")
    time.sleep(0.1)  # Force Streamlit to flush this UI to the browser before blocking
    st.session_state.imports_done = True

# ── Heavy Imports ────────────────────────────────────────────────────────────
import datetime
import logging
import plotly.graph_objects as go
from backend import (
    fetch_latest_actuals,
    load_all_models,
    build_feature_vector,
    run_predictions,
)

boot_box.empty()

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pearls_aqi.dashboard")

# ── EPA Constants ─────────────────────────────────────────────────────────────
AQI_BINS = [0, 51, 101, 151, 201, 301, 9999]
AQI_LABELS = [
    "Good", "Moderate", "Unhealthy for Sensitive Groups",
    "Unhealthy", "Very Unhealthy", "Hazardous"
]
AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023"
}
HEALTH_RECOMMENDATIONS = {
    "Good": "Air quality is satisfactory, and air pollution poses little or no risk.",
    "Moderate": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.",
    "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
    "Unhealthy": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.",
    "Very Unhealthy": "Health alert: The risk of health effects is increased for everyone.",
    "Hazardous": "Health warning of emergency conditions: everyone is more likely to be affected."
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0e1117; }

.aqi-card {
    background: #1a1f2e;
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.aqi-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7a8099;
    margin-bottom: 6px;
}
.aqi-card-value {
    font-size: 2.4rem;
    font-weight: 600;
    color: #e8eaf6;
    font-family: 'DM Mono', monospace;
    line-height: 1.1;
}
.aqi-card-sub {
    font-size: 0.82rem;
    color: #9095a8;
    margin-top: 4px;
}
.aqi-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #0e1117;
    margin-top: 8px;
}
.health-rec {
    background: #12172a;
    border-left: 3px solid #4a5270;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #9095a8;
    margin-top: 12px;
    line-height: 1.6;
}
.model-toggle-label {
    text-align: center;
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7a8099;
    padding-top: 6px;
}
.section-divider {
    border: none;
    border-top: 1px solid #2d3348;
    margin: 28px 0;
}
.page-title {
    font-size: 2.8rem !important;
    font-weight: 600 !important;
    color: #e8eaf6 !important;
    margin: 0 !important;
}
.page-subtitle {
    font-size: 1.2rem !important;
    color: #7a8099 !important;
    margin-top: 6px !important;
    margin-bottom: 0 !important;
}
.metric-winner {
    color: #00e400;
    font-weight: 600;
}
.callout-box {
    background: #12172a;
    border: 1px solid #2d3348;
    border-left: 3px solid #00e400;
    border-radius: 8px;
    padding: 18px 24px;
    font-size: 1.05rem !important;
    color: #e8eaf6 !important;
    line-height: 1.8 !important;
    margin-top: 24px;
}
.status-footer {
    margin-top: 32px;
    padding: 12px 0 4px 0;
    border-top: 1px solid #2d3348;
    font-size: 0.78rem;
    color: #4a5270;
    line-height: 2;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED LOADERS
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_models():
    return load_all_models()

@st.cache_data(ttl=3600, show_spinner=False)
def get_actuals():
    return fetch_latest_actuals(hours=48)


# =============================================================================
# SESSION STATE — arrow toggle
# =============================================================================
MODEL_NAMES = ["XGBoost", "Ridge", "MLP"]
MODEL_KEYS  = ["xgb",     "ridge", "mlp"]

if "model_idx" not in st.session_state:
    st.session_state.model_idx = 0

def toggle_prev():
    st.session_state.model_idx = (st.session_state.model_idx - 1) % len(MODEL_NAMES)

def toggle_next():
    st.session_state.model_idx = (st.session_state.model_idx + 1) % len(MODEL_NAMES)


# =============================================================================
# HELPERS
# =============================================================================
def aqi_badge(label: str) -> str:
    color = AQI_COLORS.get(label, "#cccccc")
    return (
        f'<span class="aqi-badge" style="background-color:{color};">'
        f'{label}</span>'
    )


def get_aqi_label(value: float) -> str:
    for i in range(len(AQI_BINS) - 1):
        if AQI_BINS[i] <= value < AQI_BINS[i + 1]:
            return AQI_LABELS[i]
    return AQI_LABELS[-1]


def fmt_metric(value: float, is_winner: bool, decimals: int = 2) -> str:
    formatted = f"{value:.{decimals}f}"
    if is_winner:
        return f'<span class="metric-winner">★ {formatted}</span>'
    return formatted


# =============================================================================
# PRE-RENDER HEADER & LOADING STATE
# =============================================================================
st.markdown('<p class="page-title">🌫️ Pearls AQI Predictor &mdash; Karachi</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Real-time air quality monitoring &amp; 3-day ML forecast</p>', unsafe_allow_html=True)
st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# =============================================================================
# LOAD DATA & PREDICTIONS
# =============================================================================
if "app_initialized" not in st.session_state:
    status_box = st.empty()
    status_box.info("⏳ **Cold Start Initialization:** Connecting to DagsHub MLflow to download models. This usually takes 1-3 minutes. Please wait...")
    
    models  = get_models()
    
    status_box.info("⏳ Fetching latest 48-hour context window from MongoDB...")
    df_48h  = get_actuals()
    latest  = df_48h.iloc[-1]
    metrics = models["metrics"]
    
    status_box.info("⏳ Running multi-horizon ML inference...")
    X       = build_feature_vector(df_48h, models["feature_columns"])
    results = run_predictions(X, models)
    
    status_box.empty()
    st.session_state.app_initialized = True
else:
    models  = get_models()
    df_48h  = get_actuals()
    latest  = df_48h.iloc[-1]
    metrics = models["metrics"]
    X       = build_feature_vector(df_48h, models["feature_columns"])
    results = run_predictions(X, models)
logger.info("Dashboard data loaded. Rendering UI.")

# =============================================================================
# SECTION 1 — HEADER (Updated with Timestamp)
# =============================================================================
latest_pkt   = latest["timestamp"].tz_convert("Asia/Karachi")
last_updated = latest_pkt.strftime("%b %d, %Y — %H:%M PKT")

st.markdown(f"""
<p class="page-subtitle" style="margin-top:-30px; margin-bottom: 20px;">
    Last updated: <strong style="color:#e8eaf6">{last_updated}</strong>
</p>
""", unsafe_allow_html=True)

# =============================================================================
# SECTION 2 — CURRENT AQI + FORECAST CARDS
# =============================================================================
col_current, col_forecast = st.columns([1, 2.2], gap="large")

# ── LEFT: Current AQI ────────────────────────────────────────────────────────
with col_current:
    current_aqi   = int(latest["aqi"])
    current_label = get_aqi_label(current_aqi)

    st.markdown(f"""
    <div class="aqi-card">
        <div class="aqi-card-title">Current AQI &mdash; Live</div>
        <div class="aqi-card-value">{current_aqi}</div>
        {aqi_badge(current_label)}
        <div class="aqi-card-sub" style="margin-top:10px;">
            PM2.5 &nbsp;·&nbsp; {latest['pm2_5']} µg/m³
        </div>
        <div class="aqi-card-sub">
            Temp &nbsp;·&nbsp; {latest['temp']}°C
            &nbsp;&nbsp;
            Humidity &nbsp;·&nbsp; {latest['humidity']}%
        </div>
    </div>
    <div class="health-rec">
        💡 {HEALTH_RECOMMENDATIONS[current_label]}
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Arrow toggle + Forecast cards ─────────────────────────────────────
with col_forecast:

    t_left, t_label, t_right = st.columns([1, 5, 1])

    with t_left:
        st.button("←", key="prev_model", use_container_width=True, on_click=toggle_prev)

    with t_label:
        active_name = MODEL_NAMES[st.session_state.model_idx]
        active_idx  = st.session_state.model_idx + 1
        st.markdown(
            f'<div class="model-toggle-label">'
            f'{active_name} &nbsp;·&nbsp; {active_idx} / {len(MODEL_NAMES)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with t_right:
        st.button("→", key="next_model", use_container_width=True, on_click=toggle_next)

    st.markdown(
        """<div style="font-size:1.0rem !important; color:#7a8099; text-align:center; margin-top:-10px; margin-bottom:15px;">
        <em>Note: ML models forecast the <strong>Daily Peak AQI</strong> severity based on the 72-hour weather forecast.</em>
        </div>""",
        unsafe_allow_html=True
    )

    # Forecast cards
    active_key  = MODEL_KEYS[st.session_state.model_idx]
    preds       = results[active_key]
    day_keys    = ["day1", "day2", "day3"]
    base_date   = latest_pkt.date()

    fc1, fc2, fc3 = st.columns(3, gap="small")
    for col_obj, day_key in zip([fc1, fc2, fc3], day_keys):
        aqi_val   = preds[day_key]
        aqi_label = preds["labels"][day_key]
        offset    = day_keys.index(day_key) + 1
        day_date  = (base_date + datetime.timedelta(days=offset)).strftime("%b %d")
        day_num   = f"Day {offset}"

        with col_obj:
            st.markdown(f"""
            <div class="aqi-card">
                <div class="aqi-card-title">{day_num} &nbsp;·&nbsp; {day_date}</div>
                <div class="aqi-card-value">{aqi_val:.0f}</div>
                {aqi_badge(aqi_label)}
                <div class="aqi-card-sub">{active_name} model</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    with st.expander("🔍 View Live Open-Meteo Weather Inputs"):
        st.markdown(
            "<div style='font-size:0.9rem; color:#9095a8; margin-bottom:12px; line-height: 1.5;'>"
            "To forecast these AQI peaks, your ML models dynamically ingested the 72-hour weather forecast "
            "and combined it with rolling 24-hour pollution averages from the Feature Store."
            "</div>", unsafe_allow_html=True
        )
        w1, w2, w3 = st.columns(3)
        for w_col, d_key, offset in zip([w1, w2, w3], day_keys, [1, 2, 3]):
            with w_col:
                t = X.iloc[0][f"temp_{d_key}"]
                w = X.iloc[0][f"wind_speed_{d_key}"]
                h = X.iloc[0][f"humidity_{d_key}"]
                st.markdown(f"<strong style='color:#e8eaf6; font-size:0.9rem;'>Day {offset} Forecast</strong>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.85rem; color:#7a8099; margin-top:4px;'>🌡️ Temp: <code>{t:.1f}°C</code><br>💨 Wind: <code>{w:.1f} km/h</code><br>💧 Hum: <code>{h:.0f}%</code></div>", unsafe_allow_html=True)

# =============================================================================
# SECTION 3 — 48H HISTORICAL CHART
# =============================================================================
st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.markdown("#### 48-Hour AQI History — Karachi")

fig = go.Figure()

# AQI category bands
band_defs = [
    (0,   50,  "rgba(0,228,0,0.07)",    "Good"),
    (50,  100, "rgba(255,255,0,0.07)",  "Moderate"),
    (100, 150, "rgba(255,126,0,0.08)",  "Unhealthy for Sensitive Groups"),
    (150, 200, "rgba(255,0,0,0.08)",    "Unhealthy"),
    (200, 300, "rgba(143,63,151,0.08)", "Very Unhealthy"),
    (300, 500, "rgba(126,0,35,0.08)",   "Hazardous"),
]

for low, high, fill, band_label in band_defs:
    fig.add_hrect(
        y0=low, y1=high,
        fillcolor=fill,
        line_width=0,
        annotation_text=band_label,
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color="#4a5270",
    )

# AQI line
fig.add_trace(go.Scatter(
    x=df_48h["timestamp"],
    y=df_48h["aqi"],
    mode="lines+markers",
    name="AQI",
    line=dict(color="#7b9cff", width=2),
    marker=dict(size=4, color="#7b9cff"),
    hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y}<extra></extra>",
))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#12172a",
    font=dict(family="DM Sans", color="#9095a8"),
    margin=dict(l=0, r=120, t=10, b=0),
    height=320,
    xaxis=dict(
        gridcolor="#2d3348",
        showgrid=True,
        title=None,
    ),
    yaxis=dict(
        gridcolor="#2d3348",
        showgrid=True,
        title="AQI (US)",
        range=[0, max(df_48h["aqi"].max() + 30, 160)],
    ),
    showlegend=False,
    hovermode="x unified",
)

st.plotly_chart(fig, width="stretch")


# =============================================================================
# SECTION 4 — MODEL PERFORMANCE METRICS
# =============================================================================
st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.markdown("#### Model Performance Metrics — Test Set")

# Build winners per cell
metric_models = ["ridge", "xgb", "mlp"]
day_map = {"Day 1": "y_day1", "Day 2": "y_day2", "Day 3": "y_day3"}

# Precompute winners
winners = {}
for day_label, day_key in day_map.items():
    for met in ["mae", "rmse", "r2"]:
        vals = {m: metrics[m][day_key][met] for m in metric_models}
        if met == "r2":
            winners[(day_label, met)] = max(vals, key=vals.get)
        else:
            winners[(day_label, met)] = min(vals, key=vals.get)

# Render table as HTML
rows_html = ""
display_models = ["XGBoost", "Ridge", "MLP", "Persistence"]
model_key_map  = {"XGBoost": "xgb", "Ridge": "ridge",
                  "MLP": "mlp", "Persistence": "persistence"}

for day_label, day_key in day_map.items():
    for i, display_name in enumerate(display_models):
        mk = model_key_map[display_name]

        if mk == "persistence":
            rmse = metrics["persistence"][day_key]["rmse"]
            mae  = metrics["persistence"][day_key]["mae"]
            r2   = metrics["persistence"][day_key]["r2"]
            rmse_html = f"{rmse:.2f}"
            mae_html  = f"{mae:.2f}"
            r2_html   = f"{r2:.3f}"
        else:
            rmse = metrics[mk][day_key]["rmse"]
            mae  = metrics[mk][day_key]["mae"]
            r2   = metrics[mk][day_key]["r2"]
            rmse_html = fmt_metric(rmse, winners[(day_label, "rmse")] == mk)
            mae_html  = fmt_metric(mae,  winners[(day_label, "mae")]  == mk)
            r2_html   = fmt_metric(r2,   winners[(day_label, "r2")]   == mk, decimals=3)

        day_cell = f"<strong>{day_label}</strong>" if i == 0 else ""
        rows_html += f"<tr><td style='color:#7a8099;width:80px'>{day_cell}</td><td>{display_name}</td><td>{rmse_html}</td><td>{mae_html}</td><td>{r2_html}</td></tr>"

    # spacer row between day groups
    rows_html += "<tr><td colspan='5' style='padding:4px 0; border-bottom:1px solid #2d3348;'></td></tr>"

table_html = f"""<div style="background:#1a1f2e;border:1px solid #2d3348;border-radius:12px;padding:20px 24px;margin-top:8px;">
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;color:#e8eaf6;">
<thead>
<tr style="color:#7a8099;font-size:0.75rem;letter-spacing:0.06em;text-transform:uppercase;">
<th style="text-align:left;padding:0 0 12px 0;width:80px">Horizon</th>
<th style="text-align:left;padding:0 0 12px 0">Model</th>
<th style="text-align:left;padding:0 0 12px 0">RMSE</th>
<th style="text-align:left;padding:0 0 12px 0">MAE</th>
<th style="text-align:left;padding:0 0 12px 0">R²</th>
</tr>
</thead>
<tbody style="line-height:2;">
{rows_html}
</tbody>
</table>
</div>"""

st.markdown(table_html, unsafe_allow_html=True)

# ── Why XGBoost callout ───────────────────────────────────────────────────────
st.markdown("""
<div class="callout-box">
    <strong style="color:#00e400; font-size:1.15rem;">★ Why XGBoost is the featured model</strong><br><br>
    XGBoost wins MAE on Day 2 and Day 3 — the longer horizons where accurate
    forecasting is hardest and most valuable. It also achieves the highest R²
    on Day 3 (0.321). Ridge leads on Day 1 MAE by a margin of 0.36 AQI points,
    which is negligible in practice. All three ML models decisively beat the
    persistence baseline (negative R²), confirming that learning from features
    adds genuine predictive value.
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 5 — SYSTEM STATUS FOOTER
# =============================================================================
mongo_ts  = latest_pkt.strftime("%b %d, %H:%M PKT")
run_id    = models["run_id"]
short_id  = run_id[:14] + "..."

st.markdown(f"""
<div class="status-footer">
    🟢 &nbsp;<strong>MongoDB Feature Store</strong> connected
    &nbsp;·&nbsp; Last data point: <code>{mongo_ts}</code>
    &nbsp;&nbsp;&nbsp;
    🟢 &nbsp;<strong>DagsHub MLflow</strong> model loaded
    &nbsp;·&nbsp; Run ID: <code>{short_id}</code>
</div>
""", unsafe_allow_html=True)

logger.info("Dashboard render complete.")