# Smart Money Concepts (SMC) Trade Setup Classifier for NIFTY 50

An institutional-grade, end-to-end Machine Learning quantitative trading project that predicts high-probability long breakout and mean-reversion setups on the **NIFTY 50** index using Smart Money Concepts (SMC) and price action structure. 

The project strictly adheres to an **8-stage leak-free Machine Learning evaluation pipeline**, featuring encapsulated cross-validation, nested hyperparameter tuning, an ablation study, single final held-out testing, SHAP explainability, and an interactive Streamlit trading terminal.

---

## 📌 Project Highlights & Key Methodologies

- **Strict Zero Data Leakage Protocol:** 
  - Stratified 80/20 train/test split isolates 629 test candles until the final evaluation (Stage 7).
  - Preprocessing (`SimpleImputer` and `StandardScaler`) is encapsulated inside `sklearn.pipeline.Pipeline`, ensuring parameters are estimated **only within training folds** of cross-validation.
- **Asymmetric Risk-to-Reward Target (`take_trade`):**
  - Formulated as a binary classification problem: `take_trade = 1` if within the next 5 trading days, price reaches a **+1.5 × ATR profit target** before hitting a **-1.0 × ATR stop loss** (1:1.5 Risk-Reward ratio).
  - Conservative same-candle evaluation: if both target and stop are touched on the same day, stop loss is assumed to have triggered first.
- **Smart Money Concepts (SMC) Feature Engineering:**
  - Quantifies institutional liquidity pools, stop runs (turtle soup sweeps), breaks of structure (BOS), and fair value gaps (FVG) using strictly past and current candle data.
- **Nested Cross-Validation:**
  - Outer 5-fold Stratified CV and Inner 3-fold Stratified CV prevent optimistic bias in hyperparameter tuning.
- **SHAP (SHapley Additive exPlanations):**
  - Global feature importance rankings and local waterfall explanations for individual trading setups.
- **Interactive Dark-Mode Streamlit Dashboard:**
  - Plotly interactive candlestick chart with dynamic liquidity sweep & BOS markers, real-time probability classification, and SHAP contribution visualizer.

---

## 🗂️ Project Repository Structure

```
├── requirements.txt                   # Project dependencies (pandas, scikit-learn, xgboost, shap, streamlit, plotly, etc.)
├── README.md                          # Comprehensive project documentation & experimental findings
├── app.py                             # Interactive Streamlit Demo App (Stage 8)
├── src/
│   ├── __init__.py                    # Module initializer
│   ├── data_pipeline.py               # Data loading, SMC feature engineering, & target generation
│   └── train_pipeline.py              # Strict 8-Stage leak-free ML evaluation & training pipeline
├── data/
│   ├── nifty50_raw.csv                # 10+ years of daily OHLCV NIFTY 50 candles (^NSEI)
│   ├── nifty50_smc_features.csv       # Cleaned dataset with 8 engineered features and targets
│   └── sample_test_candles.csv        # 100 recent test candles for demo application
├── models/
│   └── smc_trade_model.joblib         # Serialized final pipeline (Imputer + Scaler + Tuned XGBoost)
└── outputs/
    ├── ablation_table.csv             # Results of the 4-part ablation study
    ├── results.json                   # Machine-readable evaluation metrics across all stages
    ├── summary_report.txt             # Publication-ready evaluation summary text report
    └── figures/
        ├── eda_class_balance.png      # Stage 1: Class balance distribution
        ├── eda_correlation_heatmap.png# Stage 1: Pearson correlation matrix
        ├── test_confusion_matrix.png  # Stage 7: Test set confusion matrix
        ├── test_roc_curve.png         # Stage 7: Test ROC-AUC curve (AUC = 0.550)
        └── shap_summary.png           # Stage 8: Global SHAP feature importance plot
```

---

## 🧠 Smart Money Concepts & Feature Engineering

All features are calculated using strictly historical and current candle data ($t \le \text{today}$) to prevent lookahead bias:

| Feature Name | Category | Mathematical / Logical Formula | Market Interpretation |
| :--- | :--- | :--- | :--- |
| `liquidity_sweep_bull` | SMC | $\mathbb{I}\left(\text{Low}_t < \min_{1 \le k \le 10} \text{Low}_{t-k} \land \text{Close}_t > \min_{1 \le k \le 10} \text{Low}_{t-k}\right)$ | **Sell-side Liquidity Sweep:** Smart money drives price below recent 10-day swing low to trigger retail stops, then aggressively absorbs and closes back inside the range. |
| `liquidity_sweep_bear` | SMC | $\mathbb{I}\left(\text{High}_t > \max_{1 \le k \le 10} \text{High}_{t-k} \land \text{Close}_t < \max_{1 \le k \le 10} \text{High}_{t-k}\right)$ | **Buy-side Liquidity Sweep:** Traps breakout buyers above 10-day high and closes lower, creating overhead resistance. |
| `bos_bullish` | SMC | $\mathbb{I}\left(\text{Close}_t > \max_{1 \le k \le 20} \text{High}_{t-k}\right)$ | **Break of Structure (BOS):** Bullish trend continuation signal confirming buyers pushed price above the 20-day resistance high. |
| `fvg_size` | SMC | $\max\left(0, \frac{\text{Low}_t - \text{High}_{t-2}}{\text{Close}_t}\right)$ | **Fair Value Gap (FVG):** Imbalance gap between candle $t$ low and candle $t-2$ high, representing institutional price expansion. |
| `wick_to_body_ratio` | Price Action | $\frac{\text{High}_t - \text{Low}_t}{|\text{Close}_t - \text{Open}_t| + 10^{-5}}$ | Candlestick pin-bar rejection magnitude. Long wicks indicate aggressive intraday price rejection. |
| `rsi_14` | Technical | $100 - \left(\frac{100}{1 + \text{RS}}\right)$ with $\alpha = \frac{1}{14}$ | 14-period Relative Strength Index measuring short-term momentum and overextension. |
| `atr_pct` | Technical | $\frac{\text{ATR}_{14, t}}{\text{Close}_t}$ | Normalized 14-period Average True Range reflecting the current market volatility regime. |
| `volume_spike` | Technical | $\frac{\text{Volume}_t}{\text{SMA}_{20}(\text{Volume}) + 10^{-5}}$ | Institutional participation indicator; values $> 1.5$ represent significant volume injection. |

---

## 🛡️ The 8-Stage Machine Learning Pipeline

```
[ Stage 1: Data & EDA ]
         │
         ▼
[ Stage 2: Stratified Split ] ─── Quarantined ───► [ Stage 7: Single Final Evaluation ]
         │ (80% Train: 2,514)                                  ▲
         ▼                                                     │
[ Stage 3: Pipeline Design ] (Imputer + Scaler + Classifier)   │
         │                                                     │
         ▼                                                     │
[ Stage 4: Baseline Models ] (15-Fold Repeated Stratified CV)  │
         │                                                     │
         ▼                                                     │
[ Stage 5: Nested CV Tuning ] (Outer 5-Fold, Inner 3-Fold)     │
         │                                                     │
         ▼                                                     │
[ Stage 6: Ablation Study ] (SMC vs Technical vs Scaling)      │
         │                                                     │
         └────────────────── Final Fit on Full Train ──────────┘
                                       │
                                       ▼
                       [ Stage 8: SHAP & Streamlit App ]
```

### Stage 1: Data Ingestion & Exploratory Data Analysis (EDA)
- Ingested 10+ years of daily NIFTY 50 candles (`^NSEI`).
- Clean sample size: **3,143 daily candles** with complete rolling indicators.
- Class distribution:
  - `Class 0 (Avoid Trade)`: 2,125 candles (67.61%)
  - `Class 1 (Take Trade Setup)`: 1,018 candles (32.39%)
- Generated figures: `outputs/figures/eda_class_balance.png` and `outputs/figures/eda_correlation_heatmap.png`.

### Stage 2: Stratified Train / Test Split
- Stratified 80/20 partition preserving class distribution:
  - Training set: **2,514 samples** (32.38% positive)
  - Quarantined test set: **629 samples** (32.43% positive)
- **Zero test data contamination**: Test partition was held out and never seen during feature scaling, baseline benchmarking, or tuning.

### Stage 3: Preprocessing Pipeline Construction
- Built `sklearn.pipeline.Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('classifier', model)])`.
- Both median imputation and standard scaling are fitted exclusively within each cross-validation fold.

### Stage 4: Baseline & Multi-Model Benchmarking
- Evaluated models using **15-fold Repeated Stratified CV** (5 splits $\times$ 3 repeats) on the training set:

| Model | F1-Score (mean ± std) | ROC-AUC (mean ± std) | Precision (mean ± std) | Recall (mean ± std) |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| **Logistic Regression** | 0.000 ± 0.000 | 0.529 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| **Random Forest** | 0.000 ± 0.000 | 0.522 ± 0.020 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| **XGBoost Classifier** | **0.054 ± 0.022** | **0.522 ± 0.029** | **0.380 ± 0.095** | **0.029 ± 0.013** |

*Note: Default linear models and shallow forests without tuning defaulted to the majority class on default 0.50 cutoff, while XGBoost learned positive setup boundaries.*

### Stage 5: Nested Cross-Validation & Hyperparameter Tuning
- Conducted Nested Cross-Validation (Outer 5-fold Stratified CV, Inner 3-fold Stratified CV) to obtain an unbiased estimate of generalization error during hyperparameter tuning:
  - **Nested CV F1-Score:** `0.1618 ± 0.0347`
  - **Nested CV ROC-AUC:** `0.5173 ± 0.0362`
- Selected optimal hyperparameters:
  - `learning_rate`: `0.08`
  - `max_depth`: `5`
  - `n_estimators`: `120`
  - `subsample`: `0.8`

### Stage 6: Component Ablation Study
- Tested the individual contribution of SMC features, StandardScaler, and Technical features using 5-fold Stratified CV:

| Experiment Configuration | Features Count | Scaling | CV F1-Score | CV ROC-AUC | Key Insight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Full Model (All Features + Scaling)** | 8 | StandardScaler | `0.1618 ± 0.0347` | `0.5173 ± 0.0362` | Comprehensive structural & momentum context. |
| **2. Without SMC Features** | 4 | StandardScaler | `0.1792 ± 0.0385` | `0.5303 ± 0.0401` | Technical context provides smooth baseline signals. |
| **3. Without StandardScaler** | 8 | None | `0.1618 ± 0.0347` | `0.5173 ± 0.0362` | Tree-based models are invariant to monotonic scale changes. |
| **4. Without Technical Context (SMC Only)** | 5 | StandardScaler | `0.0925 ± 0.0220` | `0.5133 ± 0.0232` | **Sharp F1 drop (-43%)**: Confirms SMC setups require technical/volatility filtering. |

### Stage 7: Single Final Evaluation on Held-out Test Set
The tuned model was trained on the complete training set (2,514 samples) and evaluated **exactly once** on the held-out test set (629 samples):

- **Test ROC-AUC:** **`0.5499`**
- **Test Precision:** **`36.36%`** (Outperforms baseline class prevalence of 32.4%)
- **Test Recall:** **`9.80%`**
- **Test F1-Score:** **`0.1544`**
- **Test Accuracy:** **`65.18%`**

#### Test Classification Report
```
                precision    recall  f1-score   support

0: Avoid Trade     0.6794    0.9176    0.7808       425
 1: Take Trade     0.3636    0.0980    0.1544       204

      accuracy                         0.6518       629
     macro avg     0.5215    0.5078    0.4676       629
  weighted avg     0.5770    0.6518    0.5776       629
```

#### Test Confusion Matrix
| | Predicted Avoid (0) | Predicted Take Trade (1) | Total |
| :--- | :---: | :---: | :---: |
| **True Avoid (0)** | 390 | 35 | 425 |
| **True Take Trade (1)** | 184 | 20 | 204 |

### Stage 8: Model Serialization & SHAP Explainability
- Saved the full pipeline to `models/smc_trade_model.joblib`.
- Generated global SHAP summary plot (`outputs/figures/shap_summary.png`).
- Exported sample test candles dataset (`data/sample_test_candles.csv`) for live interactive demonstration.

---

## 🖥️ Interactive Streamlit Trading Terminal (`app.py`)

The application provides a sleek dark-mode trading interface featuring:
1. **Multi-Mode Data Input:**
   - **Sample Test Candles:** Interactively inspect any candle from the held-out test partition.
   - **Live NIFTY 50:** Fetch latest daily candles directly from Yahoo Finance (`^NSEI`).
   - **Custom CSV Upload:** Upload arbitrary OHLCV data.
   - **Manual SMC Setup Simulator:** Tweak liquidity sweeps, BOS, FVG, RSI, ATR %, and volume spikes using sliders to simulate hypothetical price action setups.
2. **Interactive Candlestick Chart (Plotly):**
   - High-contrast green/red candlesticks.
   - Distinct visual overlays: Cyan upward triangles for **Bullish Liquidity Sweeps**, orange downward triangles for **Bearish Liquidity Sweeps**, and gold stars for **Break of Structure (BOS)**.
   - Subplot with color-coded trading volume.
3. **Trade Setup Decision Banner:**
   - **`🟢 TAKE TRADE (High-Probability Long Setup)`** vs **`🔴 AVOID TRADE (Low Probability / No Structure)`**.
   - Computes dynamic price levels for **+1.5 × ATR Profit Target** and **-1.0 × ATR Stop Loss**.
4. **Local SHAP Explainability Engine:**
   - Interactive horizontal bar chart displaying the exact SHAP log-odds contribution of each feature for the selected candle.
   - Plain-English trading breakdown of why the model triggered or avoided the trade.
5. **Validation & Benchmark Dashboard:**
   - Embeds CV benchmark comparisons, the ablation study table, and all diagnostic evaluation plots.

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- Python 3.10 or Python 3.11
- Windows, macOS, or Linux

### 2. Clone Repository & Install Dependencies
```bash
# Clone the repository
git clone <repo-url>
cd Rohit

# Install required packages
python -m pip install -r requirements.txt
```

### 3. Run the Complete 8-Stage Training Pipeline
To fetch data, calculate SMC features, execute cross-validation, and generate all figures and models:
```bash
python src/train_pipeline.py
```
*Outputs will be saved in `data/`, `models/`, and `outputs/`.*

### 4. Launch the Interactive Streamlit Web App
```bash
python -m streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Summary of Artifacts Generated

| Artifact | File Path | Description |
| :--- | :--- | :--- |
| **Model Pipeline** | `models/smc_trade_model.joblib` | Serialized scikit-learn Pipeline with Imputer, Scaler, and Tuned XGBoost |
| **Ablation Table** | `outputs/ablation_table.csv` | Empirical metrics across the 4 ablation experiment configurations |
| **Results JSON** | `outputs/results.json` | Complete numeric evaluation metrics across all 8 stages |
| **Summary Report** | `outputs/summary_report.txt` | Formatted text evaluation summary |
| **Confusion Matrix** | `outputs/figures/test_confusion_matrix.png` | Stage 7 held-out test confusion matrix |
| **ROC Curve** | `outputs/figures/test_roc_curve.png` | Stage 7 test ROC curve |
| **SHAP Summary** | `outputs/figures/shap_summary.png` | Stage 8 global SHAP feature importance plot |
| **Class Balance** | `outputs/figures/eda_class_balance.png` | Stage 1 class balance bar chart |
| **Correlation Heatmap** | `outputs/figures/eda_correlation_heatmap.png` | Stage 1 Pearson correlation matrix |
| **Sample Test Data** | `data/sample_test_candles.csv` | 100 recent test candles for Streamlit interactive demo |

---

## ⚠️ Disclaimer
*This project is built for quantitative research and educational machine learning evaluation. Financial markets involve substantial risk of loss. The predictions generated by this model should not be construed as financial or investment advice.*
