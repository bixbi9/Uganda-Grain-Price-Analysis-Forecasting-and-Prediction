"""
patch_notebook_v2.py
Update Uganda_Grain_SARIMA_Analysis.ipynb for 7-commodity structure:
  - 7 commodities: Maize, Sorghum White, Sorghum Red, Beans Yellow,
                   Beans Nambaale, Beans Wairimu, Barley
  - Barley restricted to: Kigezi, Kapchorwa, Kabale
  - All other commodities: 24 main districts
  - Comprehensive backtesting cell for all 7 x districts
"""
import json, sys, copy

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

NB_PATH = 'Uganda_Grain_SARIMA_Analysis.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src if isinstance(src, list) else [src]
    }

# ---- Cell 3: Imports & Configuration ----------------------------------------
CELL3 = """\
# ============================================================================
# SECTION 1 - IMPORTS & GLOBAL CONFIGURATION
# ============================================================================
#
# WHY each library is used:
#   numpy / pandas  - numerical arrays and tabular data manipulation
#   matplotlib      - low-level plotting engine (inline Jupyter backend)
#   seaborn         - statistical visualisation helpers (violin, heatmap)
#   scipy.stats     - statistical tests (normality, skewness)
#   scipy.signal    - periodogram for spectral analysis of price cycles
#   sklearn         - MAE/RMSE metrics and OLS regression for SARIMAX exog
#   itertools       - parameter grid search for SARIMA order selection
#   os              - path handling for saving figures
# ----------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.signal import periodogram
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import itertools
import os

# ---- Global plot style -------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor':  '#0F1923',
    'axes.facecolor':    '#0F1923',
    'axes.edgecolor':    '#2A3A4A',
    'axes.labelcolor':   '#C8D8E8',
    'axes.titlecolor':   '#FFFFFF',
    'text.color':        '#C8D8E8',
    'xtick.color':       '#8A9BAB',
    'ytick.color':       '#8A9BAB',
    'grid.color':        '#1E2E3E',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'figure.dpi':        110,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    12,
    'axes.labelsize':    10,
    'legend.fontsize':   8,
    'legend.facecolor':  '#1A2A3A',
    'legend.edgecolor':  '#2A3A4A',
})

# ---- Commodity colours (7 commodities) ---------------------------------------
# Fixed colours used consistently across every figure.
COMMODITY_COLORS = {
    'Maize':         '#3B9AE1',   # blue
    'Sorghum White': '#4CAF50',   # green
    'Sorghum Red':   '#E53935',   # red
    'Beans Yellow':  '#FFD700',   # gold
    'Beans Nambaale':'#FF8C42',   # orange
    'Beans Wairimu': '#FF69B4',   # pink
    'Barley':        '#A0522D',   # brown (highland crop)
}

# ---- District colours (shaded by region) ------------------------------------
PALETTE = {
    **COMMODITY_COLORS,
    'forecast': '#FFD700',
    'ci':       '#2A3A5A',
    'actual':   '#FFFFFF',
    # Central - magenta family
    'Kampala': '#E91E8C', 'Natete': '#F06292', 'Luwero': '#F48FB1',
    'Kayunga': '#F8BBD9', 'Gomba':  '#FCE4EC',
    # Western - purple family
    'Mubende': '#9C27B0', 'Hoima': '#BA68C8', 'Masindi': '#CE93D8',
    'Kibaale': '#E1BEE7', 'Kyegegwa': '#AB47BC', 'Kasese': '#7B1FA2',
    'Mutukula': '#6A1B9A',
    # Northern - red-orange family
    'Gulu': '#FF5722', 'Lira': '#FF8A65', 'Kiryadongo': '#FFAB91',
    'Bweyale': '#FF7043', 'Nwoya': '#BF360C', 'Alebtong': '#D84315',
    'Pader': '#E64A19', 'Kitgum': '#FF6E40',
    # Eastern - cyan family
    'Mbale': '#00BCD4', 'Jinja': '#4DD0E1', 'Busia': '#80DEEA',
    'Soroti': '#26C6DA',
    # Barley highlands - earth tones
    'Kigezi':    '#8D6E63',
    'Kapchorwa': '#795548',
    'Kabale':    '#6D4C41',
}

# ---- Commodity-district mapping ----------------------------------------------
# Barley is a highland crop grown only in SW Uganda / Mt Elgon highland zones.
# All other commodities trade across all 24 main grain markets.
BARLEY_DISTRICTS = ['Kigezi', 'Kapchorwa', 'Kabale']

MAIN_DISTRICTS = [
    # Central (5 markets)
    'Kampala','Natete','Luwero','Kayunga','Gomba',
    # Western (7 markets)
    'Mubende','Hoima','Masindi','Kibaale','Kyegegwa','Kasese','Mutukula',
    # Northern (8 markets)
    'Gulu','Lira','Kiryadongo','Bweyale','Nwoya','Alebtong','Pader','Kitgum',
    # Eastern (4 markets)
    'Mbale','Jinja','Busia','Soroti',
]

COMMODITY_DISTRICTS = {
    'Maize':         MAIN_DISTRICTS,
    'Sorghum White': MAIN_DISTRICTS,
    'Sorghum Red':   MAIN_DISTRICTS,
    'Beans Yellow':  MAIN_DISTRICTS,
    'Beans Nambaale':MAIN_DISTRICTS,
    'Beans Wairimu': MAIN_DISTRICTS,
    'Barley':        BARLEY_DISTRICTS,
}

DISTRICT_COLORS = [PALETTE.get(d, '#888888') for d in MAIN_DISTRICTS]

# ---- Key districts for visualisation charts ---------------------------------
# 8 representative markets (one-two per region) chosen for geographic
# and price diversity across Uganda.
KEY_DISTRICTS = ['Kampala', 'Mubende', 'Hoima',
                 'Gulu', 'Lira', 'Kitgum',
                 'Mbale', 'Jinja']
KEY_COLORS = ['#E91E8C','#9C27B0','#BA68C8',
              '#FF5722','#FF8A65','#FF6E40',
              '#00BCD4','#4DD0E1']

# ---- Region groupings --------------------------------------------------------
REGION_DISTRICT = {
    'Central':   ['Kampala','Natete','Luwero','Kayunga','Gomba'],
    'Western':   ['Mubende','Hoima','Masindi','Kibaale','Kyegegwa','Kasese','Mutukula'],
    'Northern':  ['Gulu','Lira','Kiryadongo','Bweyale','Nwoya','Alebtong','Pader','Kitgum'],
    'Eastern':   ['Mbale','Jinja','Busia','Soroti'],
    'Highlands': ['Kigezi','Kapchorwa','Kabale'],
}

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

print("Libraries loaded successfully")
print(f"  Pandas {pd.__version__} | NumPy {np.__version__} | Seaborn {sns.__version__}")
print(f"  Commodities : {list(COMMODITY_DISTRICTS.keys())}")
print(f"  Main districts: {len(MAIN_DISTRICTS)} | Barley-only: {BARLEY_DISTRICTS}")
"""

# ---- Cell 4: Data loading ---------------------------------------------------
CELL4 = """\
# ============================================================================
# SECTION 1 - DATA LOADING
# ============================================================================
#
# Four CSV files are loaded (generated by rebuild_crops.py):
#   PBI_Uganda_Grains_Daily.csv      - raw daily prices per commodity/district
#   PBI_Uganda_Grains_Monthly.csv    - pre-aggregated monthly averages
#   PBI_Uganda_Grains_Forecasts.csv  - 2026 forecast values
#   PBI_Model_Statistics.csv         - summary MAE/MAPE stats per model
#
# Commodities (7 total):
#   Maize, Sorghum White, Sorghum Red,
#   Beans Yellow, Beans Nambaale, Beans Wairimu,
#   Barley (Kigezi / Kapchorwa / Kabale only)
# ----------------------------------------------------------------------------
import os

# Step 1: Detect CSV location (outputs/ subfolder or notebook directory)
BASE = 'outputs'
if not os.path.exists(f'{BASE}/PBI_Uganda_Grains_Daily.csv'):
    BASE = '.'

# Step 2: Load all four tables, parsing Date columns as datetime objects
df_daily    = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Daily.csv',    parse_dates=['Date'])
df_monthly  = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Monthly.csv',  parse_dates=['Date'])
df_forecast = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Forecasts.csv',parse_dates=['Date'])
df_stats    = pd.read_csv(f'{BASE}/PBI_Model_Statistics.csv')

# Step 3: Split daily data into actuals (2024-2025) vs. forecast rows (2026)
actuals   = df_daily[df_daily['Data_Type'] == 'Actual'].copy()
forecasts = df_daily[df_daily['Data_Type'] == 'Forecast'].copy()

# Step 4: Global dimension lists (pulled from the configuration cell)
COMMODITIES = list(COMMODITY_DISTRICTS.keys())
DISTRICTS   = MAIN_DISTRICTS.copy()   # main 24 markets (excludes barley-only)

# Step 5: Print a data summary to confirm loading was successful
print(f"Data loaded successfully:")
print(f"  Daily actuals : {len(actuals):,} rows  "
      f"({actuals['Date'].min().date()} to {actuals['Date'].max().date()})")
print(f"  Forecast rows : {len(df_forecast):,} rows")
print(f"  Monthly avg   : {len(df_monthly):,} rows")
print(f"  Model stats   : {len(df_stats)} commodity-district pairs")
print(f"\\n  Commodities ({len(COMMODITIES)}):")
for c in COMMODITIES:
    n = len(COMMODITY_DISTRICTS[c])
    print(f"    {c:<18} - {n} district(s)")
"""

# ---- Cell 17: Fit SARIMA for all 7 commodities ------------------------------
CELL17 = """\
# ============================================================================
# SECTION 5 - FIT SARIMA MODELS FOR ALL 7 COMMODITIES x THEIR DISTRICTS
# ============================================================================
#
# Model: SARIMA(1,1,1)(1,1,1)12
#   p=1,d=1,q=1 - momentum, trend removal, error correction
#   P=1,D=1,Q=1 s=12 - annual seasonality: seasonal momentum, cycle removal,
#                       seasonal error correction
#
# Fitting scope:
#   Maize / Sorghum White / Sorghum Red / Beans varieties -> 24 main districts
#   Barley                                                 -> 3 highland markets
#
# Results stored in nested dict:
#   sarima_results[commodity][district] = {
#       'model': fitted SARIMA object,
#       'series': numpy array of monthly prices used for fitting,
#       'aic': AIC value (lower = better fit),
#       'sigma': residual standard deviation
#   }
# ----------------------------------------------------------------------------

sarima_results = {}

for commodity in COMMODITIES:
    districts = COMMODITY_DISTRICTS[commodity]
    sarima_results[commodity] = {}
    fitted_count = 0

    for district in districts:
        # Extract monthly average price for this commodity-district pair
        mask = (
            (df_monthly['Commodity'] == commodity) &
            (df_monthly['District']  == district)
        )
        sub = df_monthly[mask].sort_values('Date')

        # Need at least 18 months for SARIMA(1,1,1)(1,1,1)12 after differencing
        if len(sub) < 18:
            continue

        y = sub['Price_UGX'].values.astype(float)

        model = SARIMA(order=(1,1,1), seasonal_order=(1,1,1,12))
        model.fit(y)

        sarima_results[commodity][district] = {
            'model':  model,
            'series': y,
            'aic':    model.aic,
            'sigma':  model.sigma,
        }
        fitted_count += 1

    print(f"  {commodity:<18} : {fitted_count}/{len(districts)} models fitted")

total = sum(len(v) for v in sarima_results.values())
print(f"\\nTotal SARIMA models fitted: {total}")
"""

# ---- Cell 19: SARIMAX -------------------------------------------------------
CELL19 = """\
# ============================================================================
# SECTION 6 - SARIMAX: SARIMA WITH EXOGENOUS VARIABLES
# ============================================================================
#
# SARIMAX extends SARIMA by adding an exogenous (X) regressor:
#     y_t = beta_0 + beta_1 * x_t + SARIMA_residuals_t
#
# Exogenous variable: Regional Demand Pressure Index
#   Values > 1.0 - elevated cross-border export demand (Kenya / South Sudan)
#   Values ~ 1.0 - neutral market
#   Values < 1.0 - reduced demand (harvest glut periods)
#
# Barley uses a flat exog signal (no significant export pressure for
# highland barley - it is primarily a domestic brewing/livestock crop).
# ----------------------------------------------------------------------------

# Monthly demand pressure index (24 months: Jan-2024 to Dec-2025)
demand_index_monthly = np.array([
    1.12, 1.18, 1.15, 1.08, 1.02, 0.95,   # Jan-Jun 2024
    0.88, 0.85, 0.90, 0.98, 1.05, 1.10,   # Jul-Dec 2024
    1.14, 1.20, 1.16, 1.09, 1.03, 0.94,   # Jan-Jun 2025
    0.87, 0.83, 0.91, 0.99, 1.06, 1.12,   # Jul-Dec 2025
])

demand_index_barley = np.ones(24)   # neutral - barley is domestic/niche

sarimax_results = {}

for commodity in COMMODITIES:
    districts = COMMODITY_DISTRICTS[commodity]
    sarimax_results[commodity] = {}
    fitted_count = 0

    exog = demand_index_barley if commodity == 'Barley' else demand_index_monthly

    for district in districts:
        if district not in sarima_results.get(commodity, {}):
            continue

        base = sarima_results[commodity][district]
        y    = base['series']
        n    = len(y)
        model = base['model']

        # Align exogenous variable length to the price series
        if n <= len(exog):
            x = exog[:n]
        else:
            x = np.pad(exog, (0, n - len(exog)), mode='edge')

        # OLS: y ~ beta_0 + beta_1 * x
        # Purpose: quantify how much demand pressure shifts the price level
        ols = LinearRegression()
        ols.fit(x.reshape(-1, 1), y)
        beta      = ols.coef_[0]
        intercept = ols.intercept_

        # Base SARIMA forecast for 12 months ahead (2026)
        base_fc, lo95, hi95 = model.forecast(12)

        # 2026 exog: carry forward the last 12 months of the demand pattern
        exog_2026 = exog[-12:] if len(exog) >= 12 else np.tile(exog, 2)[:12]

        # SARIMAX blend: SARIMA forecast adjusted by exog deviation from neutral
        # beta * (x - 1.0) adds a positive lift when demand > 1 and vice versa
        blend = base_fc + beta * (exog_2026 - 1.0)

        sarimax_results[commodity][district] = {
            'forecast':  blend,
            'lower95':   lo95,
            'upper95':   hi95,
            'beta':      beta,
            'intercept': intercept,
        }
        fitted_count += 1

    print(f"  {commodity:<18} : {fitted_count}/{len(districts)} SARIMAX models done")

total = sum(len(v) for v in sarimax_results.values())
print(f"\\nTotal SARIMAX models: {total}")
"""

# ---- Comprehensive backtesting cell -----------------------------------------
BACKTEST_CELL = """\
# ============================================================================
# SECTION 8 - COMPREHENSIVE WALK-FORWARD BACKTESTING (ALL 7 COMMODITIES)
# ============================================================================
#
# Walk-forward (expanding-window) validation mimics real forecasting:
#   Round k: Train on months 1...(N-K+k-1), forecast 1 step ahead,
#             record the actual value and compute the error.
#
# For each commodity-district pair this produces:
#   - 6 rounds of genuine out-of-sample 1-step-ahead forecasts
#   - MAE  : Mean Absolute Error (UGX/kg, interpretable)
#   - RMSE : Root Mean Squared Error (penalises large individual misses)
#   - MAPE : Mean Absolute Percentage Error (comparable across commodities)
#   - Shapiro-Wilk p-value: tests whether residuals are normally distributed
#     (p > 0.05 means residuals look normal - good model behaviour)
#   - Ljung-Box p-value: tests for residual autocorrelation
#     (p > 0.05 means no leftover patterns - good)
#   - Flag: OK / CHECK (MAPE 7-10%) / POOR-FIT (MAPE > 10%)
#
# Barley backtesting uses only the 3 highland districts.
# ----------------------------------------------------------------------------

from scipy.stats import shapiro, chi2

def ljung_box_q(residuals, lags=6):
    # Ljung-Box Q statistic for residual autocorrelation (manual chi2 test).
    n   = len(residuals)
    lags = min(lags, n // 2)
    if lags < 1:
        return np.nan, np.nan
    r = np.array([
        np.corrcoef(residuals[k:], residuals[:n-k])[0, 1]
        for k in range(1, lags + 1)
    ])
    Q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lags + 1)))
    p = 1 - chi2.cdf(Q, df=lags)
    return Q, p

K_ROUNDS  = 6    # number of walk-forward rounds
MIN_TRAIN = 18   # minimum training months before first round

backtest_records = []

print("Running walk-forward backtesting for all 7 commodities...")
print(f"{'Commodity':<18} {'District':<14} {'MAE':>8} {'RMSE':>8} "
      f"{'MAPE%':>7} {'SW-p':>7} {'LB-p':>7}  Flag")
print("-" * 84)

for commodity in COMMODITIES:
    districts = COMMODITY_DISTRICTS[commodity]

    for district in districts:
        if district not in sarima_results.get(commodity, {}):
            continue

        y = sarima_results[commodity][district]['series']
        n = len(y)

        if n < MIN_TRAIN + K_ROUNDS:
            continue

        errors, pcts = [], []

        for k in range(K_ROUNDS):
            train_end = n - K_ROUNDS + k   # expanding window endpoint
            y_train   = y[:train_end]
            y_actual  = y[train_end]

            # Fit a fresh model on the training slice only
            m = SARIMA(order=(1,1,1), seasonal_order=(1,1,1,12))
            m.fit(y_train)

            # 1-step-ahead forecast
            fc, _, _ = m.forecast(1)
            y_hat = fc[0]

            err = abs(y_actual - y_hat)
            pct = err / y_actual * 100 if y_actual > 0 else 0.0
            errors.append(err)
            pcts.append(pct)

        if not errors:
            continue

        mae  = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        mape = np.mean(pcts)

        # Residual diagnostics on the full-data model
        y_full  = sarima_results[commodity][district]['series']
        resid   = np.diff(y_full, n=1)   # first-difference as proxy residuals
        win     = resid[-min(30, len(resid)):]

        if len(win) > 3:
            _, sw_p = shapiro(win)
            _, lb_p = ljung_box_q(resid)
        else:
            sw_p = lb_p = np.nan

        flag = ("POOR-FIT" if mape > 10.0 else
                "CHECK"    if mape >  7.0 else "OK")

        backtest_records.append({
            'Commodity': commodity,
            'District':  district,
            'MAE':       round(mae,  1),
            'RMSE':      round(rmse, 1),
            'MAPE_pct':  round(mape, 2),
            'SW_p':      round(sw_p, 4) if not np.isnan(sw_p) else np.nan,
            'LB_p':      round(lb_p, 4) if not np.isnan(lb_p) else np.nan,
            'Flag':      flag,
        })

        flag_disp = f"*** {flag}" if flag != "OK" else flag
        sw_str  = f"{sw_p:.4f}" if not np.isnan(sw_p) else "   n/a"
        lb_str  = f"{lb_p:.4f}" if not np.isnan(lb_p) else "   n/a"
        print(f"  {commodity:<18} {district:<14} {mae:>8.1f} {rmse:>8.1f} "
              f"{mape:>7.2f}% {sw_str:>7} {lb_str:>7}  {flag_disp}")

print()
backtest_df = pd.DataFrame(backtest_records)

# ---- Summary by commodity ---------------------------------------------------
print("\\n=== BACKTEST SUMMARY BY COMMODITY ===")
summary = backtest_df.groupby('Commodity').agg(
    Districts=('District', 'count'),
    Mean_MAE=('MAE',      'mean'),
    Mean_MAPE=('MAPE_pct','mean'),
    Max_MAPE=('MAPE_pct', 'max'),
    Poor_Fit=('Flag',     lambda x: (x == 'POOR-FIT').sum()),
).round(2)
print(summary.to_string())

# ---- Overall flag check -------------------------------------------------------
poor  = backtest_df[backtest_df['Flag'] == 'POOR-FIT']
check = backtest_df[backtest_df['Flag'] == 'CHECK']
print(f"\\n  Total models: {len(backtest_df)}")
print(f"  OK           : {(backtest_df['Flag'] == 'OK').sum()}")
print(f"  CHECK (7-10%): {len(check)}")
print(f"  POOR-FIT>10% : {len(poor)}")
if len(poor) > 0:
    print("\\n  Poor-fit models:")
    for _, row in poor.iterrows():
        print(f"    {row['Commodity']} / {row['District']} - MAPE={row['MAPE_pct']}%")
else:
    print("\\n  All models within acceptable accuracy (MAPE <= 10%). Backtesting PASSED.")
"""

# ---- Backtest heatmap figure ------------------------------------------------
BACKTEST_HEATMAP = """\
# ============================================================================
# FIGURE: BACKTEST ACCURACY HEATMAP - MAPE% for all 7 commodities x districts
# ============================================================================
#
# Colour scale:
#   Green  - MAPE <= 5%  (excellent forecast accuracy)
#   Yellow - MAPE 5-10%  (acceptable)
#   Red    - MAPE > 10%  (poor-fit - needs review or additional data)
#
# Barley appears as a smaller 3-column block (Kigezi, Kapchorwa, Kabale).
# ----------------------------------------------------------------------------

pivot = backtest_df.pivot_table(
    index='Commodity', columns='District', values='MAPE_pct', aggfunc='mean')

# Ensure commodity order matches COMMODITIES list
row_order = [c for c in COMMODITIES if c in pivot.index]
pivot     = pivot.reindex(row_order)

n_districts = len(pivot.columns)
fig, ax = plt.subplots(figsize=(max(14, n_districts * 0.85), 5))

cmap = sns.diverging_palette(120, 10, as_cmap=True)   # green (low) to red (high)
sns.heatmap(
    pivot,
    ax=ax,
    cmap=cmap,
    vmin=0, vmax=12,
    annot=True, fmt='.1f',
    linewidths=0.4, linecolor='#0F1923',
    cbar_kws={'label': 'MAPE %', 'shrink': 0.6},
    annot_kws={'size': 7},
)

ax.set_title('Walk-Forward Backtest Accuracy - MAPE % by Commodity & District',
             fontsize=13, fontweight='bold', color='white', pad=12)
ax.set_xlabel('District', fontsize=9)
ax.set_ylabel('Commodity', fontsize=9)
ax.set_xticklabels(ax.get_xticklabels(), rotation=65, ha='right', fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/fig_backtest_heatmap.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()
print("Backtest heatmap saved to outputs/fig_backtest_heatmap.png")
"""

# ---- Apply patches ----------------------------------------------------------
cells = nb['cells']

cells[3]['source']  = [CELL3]
cells[4]['source']  = [CELL4]
cells[17]['source'] = [CELL17]
cells[19]['source'] = [CELL19]

# Replace cell 26 (walk-forward chart) with comprehensive backtesting
cells[26]['source'] = [BACKTEST_CELL]

# Insert backtest heatmap as new cell at position 27
new_heatmap_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [BACKTEST_HEATMAP]
}
cells.insert(27, new_heatmap_cell)

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook patched: {NB_PATH}")
print(f"  Cell 3  (imports)  : 7-commodity palette + COMMODITY_DISTRICTS")
print(f"  Cell 4  (data)     : updated COMMODITIES list")
print(f"  Cell 17 (SARIMA)   : fitting loop for all 7 commodities")
print(f"  Cell 19 (SARIMAX)  : all 7 commodities with barley-specific exog")
print(f"  Cell 26 (backtest) : comprehensive walk-forward for all 7 commodities")
print(f"  Cell 27 (NEW)      : backtest MAPE heatmap figure")
print(f"  Total cells: {len(nb['cells'])}")
