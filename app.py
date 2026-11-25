"""
Smart Money Concepts (SMC) Trade Setup Classifier - NIFTY 50
Interactive Streamlit Web Application (Stage 8)
Visualizes Price Action, Predicts Setup Probability, and Explains Decisions with SHAP.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
import shap
import yfinance as yf

from src.data_pipeline import (
    calculate_smc_features,
    FEATURE_COLUMNS,
    SMC_FEATURE_COLUMNS,
    TECHNICAL_FEATURE_COLUMNS,
)

# -----------------------------------------------------------------------------
# App Configuration & Custom Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SMC Trade Setup Classifier | NIFTY 50",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark-mode trading terminal styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0e14;
        color: #e6edf3;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.82rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .badge-take {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid #2ecc71;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        width: 100%;
        text-align: center;
    }
    .badge-avoid {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid #e74c3c;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        width: 100%;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model & Asset Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    model_path = "models/smc_trade_model.joblib"
    if not os.path.exists(model_path):
        st.error(f"Model artifact '{model_path}' not found! Run training pipeline first.")
        return None
    return joblib.load(model_path)


@st.cache_data
def load_cached_test_candles():
    path = "data/sample_test_candles.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        return df
    return None


@st.cache_data
def load_pipeline_results():
    path = "outputs/results.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def fetch_live_nifty(period: str = "6mo"):
    try:
        raw = yf.download("^NSEI", period=period, interval="1d", auto_adjust=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        rename_map = {c: c.capitalize() for c in raw.columns}
        raw.rename(columns=rename_map, inplace=True)
        featured = calculate_smc_features(raw)
        return featured.dropna(subset=FEATURE_COLUMNS)
    except Exception as e:
        st.sidebar.error(f"Failed to fetch live NIFTY 50 data: {e}")
        return None


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.title("🎯 Smart Money Concepts (SMC) Trade Setup Classifier")
st.caption(
    "NIFTY 50 Quantitative Price Action System | Leak-Free 8-Stage Machine Learning Pipeline | SHAP Explainability"
)

pipeline = load_trained_model()
results_data = load_pipeline_results()

if pipeline is None:
    st.error("Model pipeline could not be loaded. Please ensure 'python src/train_pipeline.py' has finished.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Setup Controls
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Setup & Data Source")

mode = st.sidebar.radio(
    "Select Operating Mode:",
    [
        "📊 Sample Test Candles (Held-Out Data)",
        "🌐 Live NIFTY 50 (^NSEI)",
        "📁 Upload Custom OHLCV CSV",
        "🎛️ Manual SMC Setup Simulator",
    ],
)

decision_threshold = st.sidebar.slider(
    "Decision Probability Threshold:",
    min_value=0.20,
    max_value=0.70,
    value=0.35,
    step=0.01,
    help="Classification threshold for triggering 'TAKE TRADE'. Lower thresholds prioritize high recall for asymmetric 1:1.5 Risk-Reward setups.",
)

df_active = None
candle_date = None
current_features = None
current_close = 24000.0
current_atr = 250.0

if mode == "📊 Sample Test Candles (Held-Out Data)":
    df_active = load_cached_test_candles()
    if df_active is not None and not df_active.empty:
        st.sidebar.success(f"Loaded {len(df_active)} held-out test candles.")
        selected_date_str = st.sidebar.selectbox(
            "Select Candle to Classify:",
            options=[str(d.date()) if hasattr(d, "date") else str(d) for d in df_active.index[::-1]],
            index=0,
        )
        candle_date = selected_date_str
        target_idx = [i for i, d in enumerate(df_active.index) if str(d).startswith(selected_date_str)][0]
        candle_row = df_active.iloc[target_idx]
        current_features = candle_row[FEATURE_COLUMNS].to_frame().T
        current_close = float(candle_row["Close"])
        current_atr = float(candle_row.get("atr_14", current_close * candle_row.get("atr_pct", 0.012)))

elif mode == "🌐 Live NIFTY 50 (^NSEI)":
    period = st.sidebar.selectbox("Fetch Period:", ["3mo", "6mo", "1y", "2y"], index=1)
    with st.spinner("Fetching live NIFTY 50 candles..."):
        df_active = fetch_live_nifty(period=period)
    if df_active is not None and not df_active.empty:
        st.sidebar.success(f"Fetched {len(df_active)} candles for ^NSEI.")
        selected_date_str = st.sidebar.selectbox(
            "Select Candle to Classify:",
            options=[str(d.date()) if hasattr(d, "date") else str(d) for d in df_active.index[::-1]],
            index=0,
        )
        candle_date = selected_date_str
        target_idx = [i for i, d in enumerate(df_active.index) if str(d).startswith(selected_date_str)][0]
        candle_row = df_active.iloc[target_idx]
        current_features = candle_row[FEATURE_COLUMNS].to_frame().T
        current_close = float(candle_row["Close"])
        current_atr = float(candle_row.get("atr_14", current_close * candle_row.get("atr_pct", 0.012)))

elif mode == "📁 Upload Custom OHLCV CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV (must contain Date, Open, High, Low, Close, Volume):", type=["csv"])
    if uploaded is not None:
        try:
            up_df = pd.read_csv(uploaded)
            if "Date" in up_df.columns:
                up_df["Date"] = pd.to_datetime(up_df["Date"])
                up_df.set_index("Date", inplace=True)
            rename_map = {c: c.capitalize() for c in up_df.columns}
            up_df.rename(columns=rename_map, inplace=True)
            df_active = calculate_smc_features(up_df).dropna(subset=FEATURE_COLUMNS)
            st.sidebar.success(f"Processed {len(df_active)} custom candles.")
            selected_date_str = st.sidebar.selectbox(
                "Select Candle to Classify:",
                options=[str(d.date()) if hasattr(d, "date") else str(d) for d in df_active.index[::-1]],
                index=0,
            )
            candle_date = selected_date_str
            target_idx = [i for i, d in enumerate(df_active.index) if str(d).startswith(selected_date_str)][0]
            candle_row = df_active.iloc[target_idx]
            current_features = candle_row[FEATURE_COLUMNS].to_frame().T
            current_close = float(candle_row["Close"])
            current_atr = float(candle_row.get("atr_14", current_close * candle_row.get("atr_pct", 0.012)))
        except Exception as e:
            st.sidebar.error(f"Error processing CSV: {e}")

else:  # Manual SMC Setup Simulator
    st.sidebar.subheader("Adjust Setup Parameters")
    sim_liq_bull = st.sidebar.selectbox("Liquidity Sweep Bullish (Sell-Side Sweep):", [1, 0], index=0)
    sim_liq_bear = st.sidebar.selectbox("Liquidity Sweep Bearish (Buy-Side Sweep):", [0, 1], index=0)
    sim_bos = st.sidebar.selectbox("Break of Structure (BOS Bullish):", [1, 0], index=0)
    sim_fvg = st.sidebar.slider("Fair Value Gap Size (%):", 0.0, 0.04, 0.008, 0.001)
    sim_wick = st.sidebar.slider("Wick-to-Body Ratio:", 0.2, 8.0, 2.5, 0.1)
    sim_rsi = st.sidebar.slider("RSI (14-Period):", 10.0, 90.0, 42.0, 1.0)
    sim_atr_pct = st.sidebar.slider("ATR % of Price:", 0.005, 0.035, 0.013, 0.001)
    sim_vol_spike = st.sidebar.slider("Volume Spike Ratio (vs 20d MA):", 0.3, 4.0, 1.8, 0.1)

    current_close = st.sidebar.number_input("Hypothetical NIFTY Price (₹):", value=25000.0, step=100.0)
    current_atr = current_close * sim_atr_pct

    current_features = pd.DataFrame(
        [
            {
                "liquidity_sweep_bull": sim_liq_bull,
                "liquidity_sweep_bear": sim_liq_bear,
                "bos_bullish": sim_bos,
                "fvg_size": sim_fvg,
                "wick_to_body_ratio": sim_wick,
                "rsi_14": sim_rsi,
                "atr_pct": sim_atr_pct,
                "volume_spike": sim_vol_spike,
            }
        ]
    )
    candle_date = "Interactive Simulated Setup"

# -----------------------------------------------------------------------------
# Prediction & Inference
# -----------------------------------------------------------------------------
if current_features is None:
    st.warning("Please choose or upload a valid dataset to begin.")
    st.stop()

# Model prediction
prob_positive = float(pipeline.predict_proba(current_features)[0, 1])
is_take_trade = prob_positive >= decision_threshold

tp_target = current_close + 1.5 * current_atr
sl_target = current_close - 1.0 * current_atr
risk_amount = 1.0 * current_atr
reward_amount = 1.5 * current_atr

# -----------------------------------------------------------------------------
# Top Display: Live Setup Banner & Metric Cards
# -----------------------------------------------------------------------------
st.write("")
col_banner, col_kpi1, col_kpi2, col_kpi3 = st.columns([2.5, 1, 1, 1])

with col_banner:
    if is_take_trade:
        st.markdown(
            f"""
            <div class="badge-take">
                🟢 TAKE TRADE (High-Probability Long Setup)<br>
                <span style="font-size: 0.95rem; font-weight: 500;">
                    Model Confidence: {prob_positive:.1%} | Cutoff Threshold: {decision_threshold:.1%}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="badge-avoid">
                🔴 AVOID TRADE (Low Probability / No Structure)<br>
                <span style="font-size: 0.95rem; font-weight: 500;">
                    Model Confidence: {prob_positive:.1%} | Cutoff Threshold: {decision_threshold:.1%}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_kpi1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">NIFTY 50 Close</div>
            <div class="metric-value">₹{current_close:,.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Target (+1.5 ATR)</div>
            <div class="metric-value" style="color: #2ecc71;">₹{tp_target:,.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Stop Loss (-1.0 ATR)</div>
            <div class="metric-value" style="color: #e74c3c;">₹{sl_target:,.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# -----------------------------------------------------------------------------
# Navigation Tabs: Chart & Prediction | Model Explainability | Evaluation Pipeline
# -----------------------------------------------------------------------------
tab_chart, tab_shap, tab_eval = st.tabs(
    ["📈 Price Action & Candlestick Chart", "🔍 SHAP Explainability (Why this Trade?)", "🏆 8-Stage Pipeline & Benchmarks"]
)

# =============================================================================
# TAB 1: Price Action & Candlestick Chart
# =============================================================================
with tab_chart:
    st.subheader(f"NIFTY 50 Candlestick Chart & SMC Overlays ({candle_date})")

    if df_active is not None and not df_active.empty:
        # Display the most recent 60 candles leading up to the selected date
        plot_df = df_active.copy()
        if candle_date != "Interactive Simulated Setup":
            # Filter up to selected date
            end_loc = [i for i, d in enumerate(plot_df.index) if str(d).startswith(candle_date)][0]
            start_loc = max(0, end_loc - 60)
            plot_df = plot_df.iloc[start_loc : end_loc + 1]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=("NIFTY 50 Daily OHLC with SMC Structure", "Trading Volume"),
            row_width=[0.22, 0.78],
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name="NIFTY 50",
                increasing_line_color="#2ecc71",
                decreasing_line_color="#e74c3c",
            ),
            row=1,
            col=1,
        )

        # Overlay Bullish Liquidity Sweeps
        bull_sweeps = plot_df[plot_df["liquidity_sweep_bull"] == 1]
        if not bull_sweeps.empty:
            fig.add_trace(
                go.Scatter(
                    x=bull_sweeps.index,
                    y=bull_sweeps["Low"] * 0.996,
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=12, color="#00bcd4"),
                    text=["Bull Sweep"] * len(bull_sweeps),
                    textposition="bottom center",
                    textfont=dict(size=9, color="#00bcd4"),
                    name="Liquidity Sweep (Bull)",
                ),
                row=1,
                col=1,
            )

        # Overlay Bearish Liquidity Sweeps
        bear_sweeps = plot_df[plot_df["liquidity_sweep_bear"] == 1]
        if not bear_sweeps.empty:
            fig.add_trace(
                go.Scatter(
                    x=bear_sweeps.index,
                    y=bear_sweeps["High"] * 1.004,
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", size=12, color="#ff9800"),
                    text=["Bear Sweep"] * len(bear_sweeps),
                    textposition="top center",
                    textfont=dict(size=9, color="#ff9800"),
                    name="Liquidity Sweep (Bear)",
                ),
                row=1,
                col=1,
            )

        # Overlay Break of Structure (BOS)
        bos_candles = plot_df[plot_df["bos_bullish"] == 1]
        if not bos_candles.empty:
            fig.add_trace(
                go.Scatter(
                    x=bos_candles.index,
                    y=bos_candles["High"] * 1.002,
                    mode="markers",
                    marker=dict(symbol="star", size=10, color="#f1c40f"),
                    name="Break of Structure (BOS)",
                ),
                row=1,
                col=1,
            )

        # Volume bars
        colors = ["#2ecc71" if c >= o else "#e74c3c" for c, o in zip(plot_df["Close"], plot_df["Open"])]
        fig.add_trace(
            go.Bar(
                x=plot_df.index,
                y=plot_df["Volume"],
                marker_color=colors,
                name="Volume",
                opacity=0.8,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40),
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Simulated setup mode: adjust sliders in the sidebar to test custom feature combinations.")

    # Trade execution details
    st.markdown("#### 📐 Risk Management & Asymmetric Payoff Parameters")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("Risk-Reward Ratio", "1 : 1.5", "Fixed Statistical Baseline")
    r_col2.metric("Target Distance", f"+₹{reward_amount:,.1f}", "+1.5 ATR")
    r_col3.metric("Stop Loss Distance", f"-₹{risk_amount:,.1f}", "-1.0 ATR")
    r_col4.metric("ATR (Volatility)", f"₹{current_atr:,.1f}", f"{current_atr/current_close:.2%} of price")


# =============================================================================
# TAB 2: SHAP Local Explainability
# =============================================================================
with tab_shap:
    st.subheader("🔍 Local Model Explainability: Why did the Model make this Decision?")
    st.caption("SHAP (SHapley Additive exPlanations) breaks down the exact marginal contribution of each indicator.")

    try:
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        classifier = pipeline.named_steps["classifier"]

        # Preprocess features through the pipeline steps
        X_imputed = imputer.transform(current_features)
        X_scaled = scaler.transform(X_imputed)
        X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

        explainer = shap.TreeExplainer(classifier)
        shap_res = explainer(X_scaled_df)
        shap_vals = shap_res.values[0]
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[0])

        # Create Plotly Horizontal Contribution Bar Chart
        contrib_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Raw Value": [current_features[col].iloc[0] for col in FEATURE_COLUMNS],
                "SHAP Value": shap_vals,
                "Impact": ["Bullish (Pushes Take Trade)" if v > 0 else "Bearish (Pushes Avoid)" for v in shap_vals],
            }
        ).sort_values("SHAP Value", ascending=True)

        bar_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in contrib_df["SHAP Value"]]

        fig_shap = go.Figure(
            go.Bar(
                x=contrib_df["SHAP Value"],
                y=contrib_df["Feature"],
                orientation="h",
                marker=dict(color=bar_colors),
                text=[f"{v:+.3f}" for v in contrib_df["SHAP Value"]],
                textposition="auto",
            )
        )
        fig_shap.update_layout(
            template="plotly_dark",
            title=f"Feature SHAP Contributions for Candle ({candle_date})",
            xaxis_title="SHAP Value (Impact on Log-Odds of Setup Success)",
            yaxis_title="Indicator / Feature",
            height=420,
            margin=dict(l=10, r=10, t=50, b=30),
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Feature interpretation cards
        st.markdown("#### 📖 Detailed Smart Money Concepts Breakdown for this Setup")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("**SMC Structural Indicators:**")
            st.markdown(
                f"- **Bullish Liquidity Sweep (`liquidity_sweep_bull` = {int(current_features['liquidity_sweep_bull'].iloc[0])}):** "
                + (
                    "🔥 **ACTIVE.** Smart money ran sell-stops beneath the 10-day swing low and closed back above the level (turtle soup reversal pattern)."
                    if current_features["liquidity_sweep_bull"].iloc[0] == 1
                    else "Inactive. No sell-side liquidity purge on this candle."
                )
            )
            st.markdown(
                f"- **Bearish Liquidity Sweep (`liquidity_sweep_bear` = {int(current_features['liquidity_sweep_bear'].iloc[0])}):** "
                + (
                    "⚠️ **ACTIVE.** Buy stops purged above 10-day high followed by a rejection close. May act as overhead resistance."
                    if current_features["liquidity_sweep_bear"].iloc[0] == 1
                    else "Clear. No overhead buy-side purge."
                )
            )
            st.markdown(
                f"- **Break of Structure (`bos_bullish` = {int(current_features['bos_bullish'].iloc[0])}):** "
                + (
                    "🚀 **ACTIVE.** Price closed above the 20-day high, confirming bullish order flow continuation."
                    if current_features["bos_bullish"].iloc[0] == 1
                    else "No breakout. Consolidating inside 20-day range."
                )
            )
            st.markdown(
                f"- **Fair Value Gap (`fvg_size` = {current_features['fvg_size'].iloc[0]:.4f}):** "
                + (
                    f"Imbalance gap of {current_features['fvg_size'].iloc[0]:.2%} detected between candle i and candle i-2. Provides bullish support."
                    if current_features["fvg_size"].iloc[0] > 0
                    else "No bullish Fair Value Gap detected."
                )
            )

        with exp_col2:
            st.markdown("**Price Action & Technical Context:**")
            st.markdown(
                f"- **Wick-to-Body Ratio (`{current_features['wick_to_body_ratio'].iloc[0]:.2f}`):** "
                + (
                    "High wick rejection indicates aggressive institutional absorption at lower price levels."
                    if current_features["wick_to_body_ratio"].iloc[0] > 2.0
                    else "Moderate/normal candlestick body."
                )
            )
            st.markdown(
                f"- **RSI 14 (`{current_features['rsi_14'].iloc[0]:.1f}`):** "
                + (
                    "Oversold territory (< 35). Favorable for mean-reversion long setups."
                    if current_features["rsi_14"].iloc[0] < 35
                    else (
                        "Overbought (> 65). Risk of exhaustion."
                        if current_features["rsi_14"].iloc[0] > 65
                        else "Neutral momentum zone (35 - 65)."
                    )
                )
            )
            st.markdown(
                f"- **ATR % (`{current_features['atr_pct'].iloc[0]:.2%}`):** "
                + "Reflects current normalized daily volatility regime."
            )
            st.markdown(
                f"- **Volume Spike (`{current_features['volume_spike'].iloc[0]:.2f}x`):** "
                + (
                    "Heavy institutional volume participation (> 1.5x of 20-day mean)."
                    if current_features["volume_spike"].iloc[0] > 1.5
                    else "Normal/average trading volume."
                )
            )

    except Exception as e:
        st.error(f"Error computing local SHAP explanation: {e}")


# =============================================================================
# TAB 3: 8-Stage Machine Learning Pipeline & Benchmarks
# =============================================================================
with tab_eval:
    st.subheader("🏆 Strict 8-Stage Machine Learning Pipeline & Validation Results")
    st.markdown(
        """
        To guarantee **zero data leakage**, the model was trained under a rigorous 8-stage protocol:
        Imputation and standard scaling are fitted strictly within CV folds; test data is held out until Stage 7;
        and hyperparameter tuning uses nested cross-validation.
        """
    )

    if results_data is not None:
        # KPI Cards
        st.markdown("#### 📊 Final Held-out Test Performance (Stage 7)")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("Test ROC-AUC", f"{results_data['stage7_final_test']['test_roc_auc']:.4f}", "Held-out Test")
        t_col2.metric("Test F1-Score", f"{results_data['stage7_final_test']['test_f1']:.4f}", "Setup Target")
        t_col3.metric("Test Precision", f"{results_data['stage7_final_test']['test_precision']:.4f}", "Positive Accuracy")
        t_col4.metric("Test Recall", f"{results_data['stage7_final_test']['test_recall']:.4f}", "Opportunity Capture")

        st.write("")
        col_bench, col_ablate = st.columns(2)

        with col_bench:
            st.markdown("#### 🥊 Model Benchmark Comparison (Stage 4)")
            st.caption("15-Fold Repeated Stratified CV on Training Set (mean ± std)")
            bench_df = pd.DataFrame(results_data.get("stage4_benchmarks", []))
            if not bench_df.empty:
                display_cols = [c for c in ["Model", "F1-Score", "ROC-AUC", "Precision", "Recall"] if c in bench_df.columns]
                st.dataframe(bench_df[display_cols], hide_index=True, use_container_width=True)

        with col_ablate:
            st.markdown("#### 🔬 Ablation Study (Stage 6)")
            st.caption("Component impact on CV generalization error")
            ablate_df = pd.DataFrame(results_data.get("stage6_ablation", []))
            if not ablate_df.empty:
                display_cols = [c for c in ["Experiment", "Num_Features", "Scaling", "CV_F1", "CV_ROC_AUC"] if c in ablate_df.columns]
                st.dataframe(ablate_df[display_cols], hide_index=True, use_container_width=True)

        st.markdown("#### 🖼️ Validation Figures & Diagnostic Plots")
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            if os.path.exists("outputs/figures/test_confusion_matrix.png"):
                st.image("outputs/figures/test_confusion_matrix.png", caption="Stage 7: Test Confusion Matrix")
            if os.path.exists("outputs/figures/eda_class_balance.png"):
                st.image("outputs/figures/eda_class_balance.png", caption="Stage 1: Class Balance Distribution")
        with fig_col2:
            if os.path.exists("outputs/figures/test_roc_curve.png"):
                st.image("outputs/figures/test_roc_curve.png", caption="Stage 7: Test ROC Curve (AUC = 0.550)")
            if os.path.exists("outputs/figures/shap_summary.png"):
                st.image("outputs/figures/shap_summary.png", caption="Stage 8: Global SHAP Summary Plot")

    else:
        st.info("Run the training pipeline to populate validation tables.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.write("---")
st.markdown(
    "<center><small style='color: #8b949e;'>Smart Money Concepts (SMC) Trade Setup Classifier | Built with Python, Scikit-Learn, XGBoost, SHAP, Plotly, & Streamlit | Risk Warning: Trading involves risk of loss.</small></center>",
    unsafe_allow_html=True,
)
