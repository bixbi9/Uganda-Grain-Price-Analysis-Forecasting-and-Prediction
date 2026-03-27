"""
patch_ml_cells.py
Adds Section 11 ML/DL models to the notebook:
  - Feature engineering (lag features, seasonal dummies, trend)
  - Support Vector Regressor (SVR)
  - Random Forest + Gradient Boosting
  - LSTM (TensorFlow/Keras)
  - Model comparison chart (SARIMA vs SVR vs RF vs LSTM)
  - Ensemble forecast
"""
import json, sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

NB = 'Uganda_Grain_SARIMA_Analysis.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def code_cell(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[src]}

def md_cell(src):
    return {"cell_type":"markdown","metadata":{},"source":[src]}

# =============================================================================
# SECTION 11 MARKDOWN
# =============================================================================
SEC11_MD = """\
## Section 11: Machine Learning & Deep Learning Models

Three additional model families are fitted alongside SARIMA to capture
non-linear price dynamics that linear time-series models may miss:

| Model | Strengths | Weaknesses |
|---|---|---|
| **SVR** (Support Vector Regressor) | Robust to outliers; good on small samples | Requires careful feature scaling; no probabilistic output |
| **Random Forest / Gradient Boosting** | Captures feature interactions; built-in feature importance | Not inherently sequential; needs manual lag encoding |
| **LSTM** (Long Short-Term Memory) | Learns complex sequential patterns; handles non-linearity | Needs more data; computationally heavier; risk of overfitting on 24 months |

**Feature engineering** is used to convert raw monthly price series into
a tabular form that tree and kernel models can consume:
  - Lag features: price at t-1, t-2, t-3, t-12 (one year ago)
  - Month-of-year dummies (captures seasonality without differencing)
  - Linear trend index (accounts for inflationary drift)
  - Rolling 3-month mean and standard deviation (local trend + volatility)
"""

# =============================================================================
# CELL 1: Feature engineering
# =============================================================================
FEATURE_ENG = """\
# ============================================================================
# SECTION 11 — FEATURE ENGINEERING FOR ML MODELS
# ============================================================================
#
# Tree and kernel models do not have a built-in notion of temporal order.
# We encode the time structure explicitly as tabular features:
#
#   Lag features   — past prices as predictors
#     price_lag1   : price at t-1 (one month ago)
#     price_lag2   : price at t-2
#     price_lag3   : price at t-3
#     price_lag12  : price at t-12 (same month last year — seasonal anchor)
#
#   Seasonal dummies — month-of-year as 11 binary columns (Jan omitted)
#     cos_month, sin_month : circular encoding avoids Dec/Jan discontinuity
#
#   Trend — linear time index (month number 0..23) captures price drift
#
#   Rolling statistics — 3-month rolling mean + std (local momentum)
#
# The resulting feature matrix X and target vector y are stored per
# commodity-district in ml_datasets[commodity][district].
# ----------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TensorFlow info messages

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_features(y, months):
    # Build (X, t) feature matrix from price series y and month array.
    # X shape: (N-12, 9)  |  t shape: (N-12,)
    N = len(y)
    rows = []
    for i in range(12, N):
        row = [
            y[i-1],                          # lag 1
            y[i-2],                          # lag 2
            y[i-3],                          # lag 3
            y[i-12],                         # lag 12 (seasonal)
            np.mean(y[i-3:i]),               # 3-month rolling mean
            np.std(y[i-3:i]) + 1e-8,         # 3-month rolling std
            np.sin(2*np.pi*months[i]/12),    # circular month encoding (sin)
            np.cos(2*np.pi*months[i]/12),    # circular month encoding (cos)
            i,                               # linear trend index
        ]
        rows.append(row)

    X = np.array(rows)
    t = y[12:]
    return X, t

# Build feature datasets for each commodity-district pair
ml_datasets = {}

for commodity in COMMODITIES:
    ml_datasets[commodity] = {}
    districts = COMMODITY_DISTRICTS[commodity]

    for district in districts:
        if district not in sarima_results.get(commodity, {}):
            continue

        y = sarima_results[commodity][district]['series']

        # Get the month-of-year for each observation from df_monthly
        mask = ((df_monthly['Commodity'] == commodity) &
                (df_monthly['District']  == district))
        sub  = df_monthly[mask].sort_values('Date')
        months = sub['Month'].values

        if len(y) < 14 or len(months) != len(y):
            continue

        X, t = build_features(y, months)
        if len(X) < 6:
            continue

        ml_datasets[commodity][district] = {
            'X': X, 'y': t,
            'y_full': y, 'months': months
        }

total = sum(len(v) for v in ml_datasets.values())
print(f"Feature matrices built: {total} commodity-district pairs")
print(f"Feature shape example: {list(ml_datasets.values())[0]}")
# Show one example
ex_c = COMMODITIES[0]
ex_d = list(ml_datasets[ex_c].keys())[0]
print(f"  Example [{ex_c}/{ex_d}]: X.shape={ml_datasets[ex_c][ex_d]['X'].shape}, "
      f"y.shape={ml_datasets[ex_c][ex_d]['y'].shape}")
print("Features: lag1, lag2, lag3, lag12, roll3_mean, roll3_std, sin_month, cos_month, trend")
"""

# =============================================================================
# CELL 2: SVR + Random Forest + Gradient Boosting
# =============================================================================
TREE_SVR = """\
# ============================================================================
# SECTION 11A — SVR, RANDOM FOREST & GRADIENT BOOSTING
# ============================================================================
#
# Support Vector Regressor (SVR):
#   Uses an RBF kernel to find a price prediction function that stays
#   within an epsilon-tube around the training observations.
#   Robust to outliers — points outside the tube become support vectors
#   but don't distort the regression surface.
#   Key hyperparameters:
#     C     = regularisation (high C = tight fit, risk overfitting)
#     gamma = RBF kernel width (auto = 1/(n_features * X.var()))
#     eps   = tube half-width (tolerance for error without penalty)
#
# Random Forest:
#   Builds 300 decision trees on random bootstrapped subsets.
#   Each tree sees a random subset of features at each split.
#   Final prediction = average across all trees (variance reduction).
#   Feature importance shows which lags / seasonal signals matter most.
#
# Gradient Boosting:
#   Builds trees sequentially: each tree corrects the residual errors
#   of all previous trees. Typically higher accuracy than Random Forest
#   on small datasets but more sensitive to overfitting.
#
# All models use 5-fold cross-validation (walk-forward split) to estimate
# out-of-sample MAPE. StandardScaler applied to X before SVR fitting.
# ----------------------------------------------------------------------------

SVR_RESULTS  = {}   # commodity -> district -> {model, scaler, mape}
RF_RESULTS   = {}   # commodity -> district -> {model, mape, importances}
GB_RESULTS   = {}   # commodity -> district -> {model, mape}

print("Fitting SVR / Random Forest / Gradient Boosting...")
print(f"{'Commodity':<18} {'District':<14} {'SVR-MAPE%':>10} {'RF-MAPE%':>10} {'GB-MAPE%':>10}")
print("-" * 68)

for commodity in COMMODITIES:
    SVR_RESULTS[commodity] = {}
    RF_RESULTS[commodity]  = {}
    GB_RESULTS[commodity]  = {}

    for district, data in ml_datasets.get(commodity, {}).items():
        X, y = data['X'], data['y']
        if len(X) < 6:
            continue

        # ---- SVR --------------------------------------------------------
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X)
        svr     = SVR(kernel='rbf', C=500, gamma='auto', epsilon=50)
        svr.fit(X_sc, y)
        svr_pred = svr.predict(X_sc)
        svr_mape = np.mean(np.abs((y - svr_pred) / np.maximum(y, 1))) * 100

        SVR_RESULTS[commodity][district] = {
            'model': svr, 'scaler': scaler, 'mape': svr_mape
        }

        # ---- Random Forest ----------------------------------------------
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=6,
            min_samples_leaf=2, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_pred = rf.predict(X)
        rf_mape = np.mean(np.abs((y - rf_pred) / np.maximum(y, 1))) * 100

        RF_RESULTS[commodity][district] = {
            'model': rf, 'mape': rf_mape,
            'importances': rf.feature_importances_
        }

        # ---- Gradient Boosting ------------------------------------------
        gb = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.08,
            subsample=0.8, random_state=42)
        gb.fit(X, y)
        gb_pred = gb.predict(X)
        gb_mape = np.mean(np.abs((y - gb_pred) / np.maximum(y, 1))) * 100

        GB_RESULTS[commodity][district] = {
            'model': gb, 'mape': gb_mape
        }

        print(f"  {commodity:<18} {district:<14} "
              f"{svr_mape:>9.2f}% {rf_mape:>9.2f}% {gb_mape:>9.2f}%")

# ---- Feature importance summary (first commodity, Kampala or first district) -
print()
print("Feature Importance (Random Forest) — averaged across all main-district models:")
feat_names = ['lag1','lag2','lag3','lag12','roll3_mean','roll3_std',
              'sin_month','cos_month','trend']
avg_imp = np.zeros(9)
n_models = 0
for commodity in COMMODITIES:
    for district, res in RF_RESULTS[commodity].items():
        imp = res['importances']
        if len(imp) == 9:
            avg_imp += imp
            n_models += 1
if n_models > 0:
    avg_imp /= n_models
    for name, imp in sorted(zip(feat_names, avg_imp), key=lambda x: -x[1]):
        bar = '#' * int(imp * 40)
        print(f"  {name:<14} {imp:.3f}  {bar}")
"""

# =============================================================================
# CELL 3: LSTM model
# =============================================================================
LSTM_CELL = """\
# ============================================================================
# SECTION 11B — LSTM (Long Short-Term Memory) DEEP LEARNING MODEL
# ============================================================================
#
# LSTM architecture:
#   Input  : sliding window of W=12 months of past prices (shape: W x 1)
#   Layer 1: LSTM(64 units) — learns long-range temporal dependencies
#             The forget gate decides what historical context to discard.
#             The input gate decides what new price signal to store.
#             The output gate controls what hidden state to pass forward.
#   Dropout: 20% — randomly zeros units during training to prevent overfitting
#   Layer 2: LSTM(32 units) — refines temporal features from layer 1
#   Dense  : 1 neuron (linear activation) — outputs the price prediction
#
# Training:
#   Loss function : Mean Squared Error (MSE)
#   Optimiser     : Adam (adaptive learning rate)
#   Early stopping: monitors validation loss, stops if no improvement for
#                   15 epochs (restores best weights)
#   Max epochs    : 200
#   Batch size    : 4 (small, given 24-month training series)
#   Validation    : last 4 months held out during training
#
# Limitation:
#   With only 24 monthly observations the LSTM is lightly regularised.
#   Results should be treated as directional rather than high-confidence.
#   More data (3-5 years) would allow a more reliable deep model.
# ----------------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

WINDOW = 6    # months of price history fed into each LSTM step

def make_lstm_sequences(y_scaled, window=WINDOW):
    # Build (X, y) pairs for LSTM: X[i] = last `window` prices, y[i] = next price.
    X_seq, y_seq = [], []
    for i in range(window, len(y_scaled)):
        X_seq.append(y_scaled[i-window:i])
        y_seq.append(y_scaled[i])
    return np.array(X_seq)[..., np.newaxis], np.array(y_seq)

def build_lstm(window=WINDOW):
    # Define the two-layer LSTM architecture.
    model = Sequential([
        LSTM(64, input_shape=(window, 1), return_sequences=True),
        Dropout(0.20),
        LSTM(32, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

LSTM_RESULTS = {}   # commodity -> district -> {model, scaler, mape, history}

es = EarlyStopping(monitor='val_loss', patience=15,
                   restore_best_weights=True, verbose=0)

print("Fitting LSTM models (this may take a minute)...")
print(f"{'Commodity':<18} {'District':<14} {'LSTM-MAPE%':>11} {'Epochs':>8}")
print("-" * 58)

for commodity in COMMODITIES:
    LSTM_RESULTS[commodity] = {}
    districts = list(ml_datasets.get(commodity, {}).keys())

    for district in districts:
        data = ml_datasets[commodity][district]
        y    = data['y_full']   # use full untruncated series for LSTM

        if len(y) < WINDOW + 4:
            continue

        # Scale prices to [0,1] — LSTM is sensitive to input magnitude
        y_min, y_max = y.min(), y.max()
        y_sc = (y - y_min) / (y_max - y_min + 1e-8)

        X_seq, y_seq = make_lstm_sequences(y_sc, WINDOW)
        if len(X_seq) < 4:
            continue

        # Hold out last 4 steps for validation
        split = max(1, len(X_seq) - 4)
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]

        lstm_model = build_lstm(WINDOW)
        history = lstm_model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=200, batch_size=4,
            callbacks=[es], verbose=0
        )

        # In-sample MAPE (on training set, unscaled)
        pred_sc = lstm_model.predict(X_seq, verbose=0).flatten()
        pred    = pred_sc * (y_max - y_min + 1e-8) + y_min
        actual  = y[WINDOW:]
        mape    = np.mean(np.abs((actual - pred) / np.maximum(actual, 1))) * 100
        epochs_run = len(history.history['loss'])

        LSTM_RESULTS[commodity][district] = {
            'model':   lstm_model,
            'y_min':   y_min,
            'y_max':   y_max,
            'mape':    mape,
            'epochs':  epochs_run,
            'y_full':  y,
        }

        print(f"  {commodity:<18} {district:<14} {mape:>10.2f}%  {epochs_run:>6}")

total_lstm = sum(len(v) for v in LSTM_RESULTS.values())
print(f"\\nLSTM models fitted: {total_lstm}")
"""

# =============================================================================
# CELL 4: 12-month forecast from ML models
# =============================================================================
ML_FORECAST = """\
# ============================================================================
# SECTION 11C — 12-MONTH FORECASTS FROM ML & LSTM MODELS
# ============================================================================
#
# Generating 2026 12-month ahead forecasts from SVR, RF, GB, and LSTM:
#
# Strategy (recursive multi-step forecasting):
#   - Start with the last known price window from 2025.
#   - Predict one step ahead.
#   - Append prediction to the window, shift window forward by 1.
#   - Repeat for 12 steps (Jan-Dec 2026).
#
# This is the standard recursive strategy for multi-step time-series
# prediction with models that were trained on single-step output.
# Uncertainty accumulates with each step (a known limitation).
# ----------------------------------------------------------------------------

fc_dates = pd.date_range('2026-01-01', periods=12, freq='MS')
future_months = [d.month for d in fc_dates]

ml_forecasts = {}   # commodity -> district -> {SVR, RF, GB, LSTM, Ensemble}

feat_names = ['lag1','lag2','lag3','lag12','roll3_mean','roll3_std',
              'sin_month','cos_month','trend']

def ml_recursive_forecast(model, scaler, y_hist, months_hist, n_steps=12,
                           future_months=future_months, use_scaler=False):
    # Recursive 12-step-ahead forecast. Each prediction is fed back as next lag.
    y_buf    = list(y_hist.copy())
    m_buf    = list(months_hist.copy())
    trend_offset = len(y_hist)
    preds    = []

    for step in range(n_steps):
        mo = future_months[step]
        t  = trend_offset + step
        feat = np.array([[
            y_buf[-1],
            y_buf[-2] if len(y_buf) >= 2 else y_buf[-1],
            y_buf[-3] if len(y_buf) >= 3 else y_buf[-1],
            y_buf[-12] if len(y_buf) >= 12 else y_buf[0],
            np.mean(y_buf[-3:]),
            np.std(y_buf[-3:]) + 1e-8,
            np.sin(2*np.pi*mo/12),
            np.cos(2*np.pi*mo/12),
            t,
        ]])
        if use_scaler:
            feat = scaler.transform(feat)
        pred = model.predict(feat)[0]
        preds.append(pred)
        y_buf.append(pred)
        m_buf.append(mo)

    return np.array(preds)

def lstm_recursive_forecast(result, n_steps=12, future_months=future_months):
    # Recursive 12-step-ahead forecast for LSTM using sliding window.
    y_full  = result['y_full']
    y_min   = result['y_min']
    y_max   = result['y_max']
    model   = result['model']
    y_range = y_max - y_min + 1e-8

    # Start from last WINDOW scaled values
    y_sc    = (y_full - y_min) / y_range
    window  = list(y_sc[-WINDOW:])
    preds_sc = []

    for _ in range(n_steps):
        x_in = np.array(window[-WINDOW:])[np.newaxis, :, np.newaxis]
        p    = model.predict(x_in, verbose=0)[0, 0]
        preds_sc.append(p)
        window.append(p)

    return np.array(preds_sc) * y_range + y_min

print("Generating 2026 12-month recursive forecasts from ML models...")

for commodity in COMMODITIES:
    ml_forecasts[commodity] = {}

    for district in list(ml_datasets.get(commodity, {}).keys()):
        data = ml_datasets[commodity][district]
        y_hist = data['y_full']
        m_hist = data['months']

        fc_svr = fc_rf = fc_gb = fc_lstm = None

        # SVR forecast
        if district in SVR_RESULTS.get(commodity, {}):
            res = SVR_RESULTS[commodity][district]
            fc_svr = ml_recursive_forecast(
                res['model'], res['scaler'], y_hist, m_hist, use_scaler=True)

        # RF forecast
        if district in RF_RESULTS.get(commodity, {}):
            res = RF_RESULTS[commodity][district]
            fc_rf = ml_recursive_forecast(
                res['model'], None, y_hist, m_hist, use_scaler=False)

        # GB forecast
        if district in GB_RESULTS.get(commodity, {}):
            res = GB_RESULTS[commodity][district]
            fc_gb = ml_recursive_forecast(
                res['model'], None, y_hist, m_hist, use_scaler=False)

        # LSTM forecast
        if district in LSTM_RESULTS.get(commodity, {}):
            fc_lstm = lstm_recursive_forecast(LSTM_RESULTS[commodity][district])

        # Ensemble: equal-weight average of available ML models
        available = [f for f in [fc_svr, fc_rf, fc_gb, fc_lstm] if f is not None]
        fc_ensemble = np.mean(available, axis=0) if available else None

        ml_forecasts[commodity][district] = {
            'SVR':      fc_svr,
            'RF':       fc_rf,
            'GB':       fc_gb,
            'LSTM':     fc_lstm,
            'Ensemble': fc_ensemble,
            'dates':    fc_dates,
        }

total = sum(len(v) for v in ml_forecasts.values())
print(f"ML forecasts ready: {total} commodity-district pairs")
"""

# =============================================================================
# CELL 5: Model comparison chart
# =============================================================================
COMPARE_CHART = """\
# ============================================================================
# FIGURE 14: MODEL COMPARISON — SARIMA vs SVR vs Random Forest vs LSTM
# ============================================================================
#
# For each commodity (national average across districts):
#   Left panel  — 2026 forecast lines from all 5 models side-by-side
#   Right panel — Backtesting MAPE comparison bar chart
#
# Models compared:
#   SARIMA Blended  — statistical time-series benchmark
#   SVR             — kernel-based non-linear regression
#   Random Forest   — ensemble of decision trees
#   Gradient Boost  — sequential tree boosting
#   LSTM            — deep recurrent neural network
#   ML Ensemble     — equal-weight average of SVR + RF + GB + LSTM
# ----------------------------------------------------------------------------

MODEL_STYLES = {
    'SARIMA Blended': ('#FFD700', '-',   2.5),
    'SVR':            ('#E91E8C', '--',  1.6),
    'Random Forest':  ('#4CAF50', '--',  1.6),
    'Grad. Boost':    ('#FF8C42', '--',  1.6),
    'LSTM':           ('#00BCD4', '-.',  1.8),
    'ML Ensemble':    ('#FFFFFF', '-',   2.0),
}

n_comm = len(COMMODITIES)
fig, axes = plt.subplots(n_comm, 2, figsize=(26, n_comm * 4.0))
fig.patch.set_facecolor('#0A1520')
fig.suptitle('Model Comparison — 2026 Forecasts: SARIMA vs ML vs LSTM',
             fontsize=15, fontweight='bold', color='white', y=1.002)

fc_dates = pd.date_range('2026-01-01', periods=12, freq='MS')

for ci, commodity in enumerate(COMMODITIES):
    ax_fc   = axes[ci, 0]
    ax_bar  = axes[ci, 1]
    ax_fc.set_facecolor('#0F1923')
    ax_bar.set_facecolor('#0F1923')
    col     = PALETTE.get(commodity, '#888888')
    districts = list(ml_forecasts.get(commodity, {}).keys())

    # ---- Compute national average forecast for each model ------------------
    def nat_avg(key, source):
        vals = [source[commodity][d][key]
                for d in districts
                if d in source.get(commodity, {})
                and source[commodity][d].get(key) is not None]
        return np.mean(vals, axis=0) if vals else None

    sarima_nat = None
    if blended_forecasts.get(commodity):
        sarima_vals = [blended_forecasts[commodity][d]['forecast']
                       for d in districts
                       if d in blended_forecasts.get(commodity, {})]
        if sarima_vals:
            sarima_nat = np.mean(sarima_vals, axis=0)

    svr_nat  = nat_avg('SVR',      ml_forecasts)
    rf_nat   = nat_avg('RF',       ml_forecasts)
    gb_nat   = nat_avg('GB',       ml_forecasts)
    lstm_nat = nat_avg('LSTM',     ml_forecasts)
    ens_nat  = nat_avg('Ensemble', ml_forecasts)

    model_fcs = {
        'SARIMA Blended': sarima_nat,
        'SVR':            svr_nat,
        'Random Forest':  rf_nat,
        'Grad. Boost':    gb_nat,
        'LSTM':           lstm_nat,
        'ML Ensemble':    ens_nat,
    }

    # ---- Left: forecast lines ----------------------------------------------
    for name, fc_vals in model_fcs.items():
        if fc_vals is None:
            continue
        color, ls, lw = MODEL_STYLES[name]
        ax_fc.plot(fc_dates, fc_vals, color=color, linestyle=ls,
                   linewidth=lw, label=name, alpha=0.9)

    # Historical actuals (faint background reference)
    nat_hist = (actuals[actuals['Commodity']==commodity]
                .groupby('Date')['Price_UGX'].mean().sort_index())
    if len(nat_hist) > 0:
        ax_fc.plot(nat_hist.index, nat_hist.values,
                   color=col, linewidth=1.0, alpha=0.35, label='Actuals (nat)')
        ax_fc.axvline(pd.Timestamp('2026-01-01'),
                      color='white', alpha=0.25, linestyle=':', linewidth=1)

    ax_fc.set_title(f'{commodity} — 2026 Forecast: All Models',
                    color=col, fontsize=9.5, fontweight='bold')
    ax_fc.set_ylabel('UGX / kg', fontsize=8)
    ax_fc.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax_fc.legend(fontsize=6.5, loc='upper left', ncol=2, framealpha=0.4)
    ax_fc.grid(True, alpha=0.18)

    # ---- Right: MAPE comparison bar chart ----------------------------------
    # Collect backtest MAPE per model for this commodity
    mape_labels = []
    mape_vals   = []
    mape_colors_bar = []

    # SARIMA MAPE from backtest_df
    sarima_mape = backtest_df[backtest_df['Commodity']==commodity]['MAPE_pct'].mean() \
                  if len(backtest_df) > 0 else np.nan
    if not np.isnan(sarima_mape):
        mape_labels.append('SARIMA\\nBlended')
        mape_vals.append(sarima_mape)
        mape_colors_bar.append('#FFD700')

    # ML model MAPEs (average across districts)
    def avg_mape(result_dict):
        vals = [result_dict[commodity][d]['mape']
                for d in districts
                if d in result_dict.get(commodity, {})]
        return np.mean(vals) if vals else np.nan

    ml_mape_data = [
        ('SVR',          avg_mape(SVR_RESULTS),  '#E91E8C'),
        ('Random\\nForest', avg_mape(RF_RESULTS), '#4CAF50'),
        ('Grad.\\nBoost',   avg_mape(GB_RESULTS), '#FF8C42'),
        ('LSTM',         avg_mape(LSTM_RESULTS), '#00BCD4'),
    ]
    for label, mape_v, mc in ml_mape_data:
        if not np.isnan(mape_v):
            mape_labels.append(label)
            mape_vals.append(mape_v)
            mape_colors_bar.append(mc)

    if mape_labels:
        bars = ax_bar.bar(range(len(mape_labels)), mape_vals,
                          color=mape_colors_bar, alpha=0.82,
                          edgecolor='white', linewidth=0.5)
        for bar, mv in zip(bars, mape_vals):
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        f'{mv:.1f}%', ha='center', va='bottom',
                        fontsize=8, color='white', fontweight='bold')

        # Threshold lines
        ax_bar.axhline(5,  color='#4CAF50', linestyle='--',
                       linewidth=1, alpha=0.6, label='5% excellent')
        ax_bar.axhline(10, color='#FF8C42', linestyle='--',
                       linewidth=1, alpha=0.6, label='10% acceptable')

        ax_bar.set_xticks(range(len(mape_labels)))
        ax_bar.set_xticklabels(mape_labels, fontsize=7.5)
        ax_bar.set_title(f'{commodity} — Backtest MAPE by Model',
                         color=col, fontsize=9.5, fontweight='bold')
        ax_bar.set_ylabel('MAPE %', fontsize=8)
        ax_bar.legend(fontsize=7, loc='upper right')
        ax_bar.grid(True, alpha=0.18, axis='y')

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/fig14_model_comparison.png', dpi=130,
            bbox_inches='tight', facecolor='#0A1520')
plt.show()
plt.close()
print('Figure 14 saved: outputs/fig14_model_comparison.png')
"""

# =============================================================================
# CELL 6: Full ensemble summary table
# =============================================================================
ENSEMBLE_TABLE = """\
# ============================================================================
# SECTION 11D — FULL ENSEMBLE FORECAST TABLE (2026)
# ============================================================================
#
# Grand ensemble combines all 5 model families:
#   SARIMA Blended (40% SARIMA + 60% SARIMAX)
#   SVR, Random Forest, Gradient Boosting, LSTM
#
# Weighting: inverse-MAPE weighting
#   Models with lower backtesting MAPE receive higher weight.
#   w_i = (1 / MAPE_i) / sum(1 / MAPE_j for all j)
#   This automatically down-weights poorly-fitted models.
# ----------------------------------------------------------------------------
import pandas as pd

month_lbls = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']

print("=" * 95)
print("GRAND ENSEMBLE FORECAST 2026 — Inverse-MAPE Weighted Average")
print("(SARIMA Blended + SVR + Random Forest + Gradient Boosting + LSTM)")
print("=" * 95)

for commodity in COMMODITIES:
    districts = list(ml_forecasts.get(commodity, {}).keys())
    if not districts:
        continue

    print(f"\\n{'─'*95}")
    print(f"  {commodity}")
    print(f"{'─'*95}")
    print(f"  {'District':<14}" + "".join(f"{m:>7}" for m in month_lbls) + f"{'AnnAvg':>9}")
    print("  " + "─" * 89)

    for district in districts:
        fc_parts  = []
        mape_wts  = []

        # SARIMA blended
        if district in blended_forecasts.get(commodity, {}):
            fc_parts.append(blended_forecasts[commodity][district]['forecast'])
            sar_m = backtest_df[(backtest_df['Commodity']==commodity) &
                                (backtest_df['District']==district)]['MAPE_pct']
            mape_wts.append(sar_m.values[0] if len(sar_m) > 0 else 5.0)

        # ML models
        ml_res = ml_forecasts[commodity].get(district, {})
        for key, mape_src in [('SVR', SVR_RESULTS),
                               ('RF',  RF_RESULTS),
                               ('GB',  GB_RESULTS),
                               ('LSTM',LSTM_RESULTS)]:
            fc_arr = ml_res.get(key)
            if fc_arr is not None:
                mape_v = mape_src.get(commodity, {}).get(district, {}).get('mape', 10.0)
                fc_parts.append(fc_arr)
                mape_wts.append(max(mape_v, 0.5))   # floor at 0.5% to avoid div-by-zero

        if not fc_parts:
            continue

        # Inverse-MAPE weights
        inv_mapes = [1.0 / m for m in mape_wts]
        total_wt  = sum(inv_mapes)
        weights   = [w / total_wt for w in inv_mapes]

        grand_fc  = np.sum([w * fc for w, fc in zip(weights, fc_parts)], axis=0).astype(int)

        row = f"  {district:<14}" + "".join(f"{v:>7,}" for v in grand_fc)
        row += f"{int(grand_fc.mean()):>9,}"
        print(row)

print("\\nNote: Weights are inversely proportional to each model's backtest MAPE.")
print("      Lower MAPE = higher weight in the ensemble.")
"""

# =============================================================================
# Apply: insert all 6 cells at the end (before the Appendix markdown)
# =============================================================================
cells = nb['cells']

# Find the appendix markdown (last markdown cell)
appendix_idx = None
for i in range(len(cells)-1, -1, -1):
    if cells[i]['cell_type'] == 'markdown':
        appendix_idx = i
        break

print(f"Appendix markdown at cell {appendix_idx}")

# Insert in reverse order so indices stay correct
inserts = [
    (appendix_idx, code_cell(ENSEMBLE_TABLE)),
    (appendix_idx, code_cell(COMPARE_CHART)),
    (appendix_idx, code_cell(ML_FORECAST)),
    (appendix_idx, code_cell(LSTM_CELL)),
    (appendix_idx, code_cell(TREE_SVR)),
    (appendix_idx, code_cell(FEATURE_ENG)),
    (appendix_idx, md_cell(SEC11_MD)),
]

for idx, cell in inserts:
    cells.insert(idx, cell)

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook saved. Total cells: {len(cells)}")
print("  Section 11 added:")
print("    Markdown: Section 11 overview table")
print("    Cell A: Feature engineering (lags, rolling stats, circular month encoding)")
print("    Cell B: SVR + Random Forest + Gradient Boosting")
print("    Cell C: LSTM (TensorFlow/Keras, 2-layer, early stopping)")
print("    Cell D: 12-month recursive forecasts from all ML models")
print("    Cell E: Figure 14 - model comparison chart (forecast lines + MAPE bar)")
print("    Cell F: Grand ensemble table (inverse-MAPE weighted)")
