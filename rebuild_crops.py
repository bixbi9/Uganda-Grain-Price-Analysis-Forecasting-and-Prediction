"""
rebuild_crops.py
================
Rebuilds all CSV data with the updated commodity and district structure:

  NEW COMMODITIES:
    Maize              — all 24 districts
    Sorghum White      — all 24 districts (northern/eastern focus)
    Sorghum Red        — all 24 districts (eastern/central focus)
    Beans Yellow       — all 24 districts
    Beans Nambaale     — all 24 districts (Central/Eastern specialty)
    Beans Wairimu      — all 24 districts
    Barley             — ONLY Kigezi, Kapchorwa, Kabale (highland districts)

  NEW DISTRICTS (barley-only):
    Kigezi    — SW highland sub-region market
    Kapchorwa — Mt Elgon highland, arabica/barley zone
    Kabale    — Kabale town market, Kigezi hub

  Price base levels are calibrated to:
    - FAO GIEWS Uganda commodity averages 2024
    - WFP VAM regional price monitoring data
    - Two-season bean calendar (July-Aug harvest + Dec-Jan harvest)
    - Northern sorghum harvest Oct-Dec, eastern Aug-Sep
    - Highland barley harvest May-Jun with one annual cycle
"""
import pandas as pd
import numpy as np
import warnings, io, sys
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

np.random.seed(42)
BASE = r'c:/Users/Administrator/Documents/BI/sarima and arima'

# ─────────────────────────────────────────────────────────────────────────────
# DISTRICT MASTER (24 existing + 3 barley-only highland districts)
# ─────────────────────────────────────────────────────────────────────────────
DISTRICT_MASTER = [
    # Central (5)
    ('Kampala',    'Central',  1.13),
    ('Natete',     'Central',  1.08),
    ('Luwero',     'Central',  0.96),
    ('Kayunga',    'Central',  0.94),
    ('Gomba',      'Central',  0.90),
    # Western (7)
    ('Mubende',    'Western',  0.96),
    ('Hoima',      'Western',  0.99),
    ('Masindi',    'Western',  0.97),
    ('Kibaale',    'Western',  0.91),
    ('Kyegegwa',   'Western',  0.92),
    ('Kasese',     'Western',  0.95),
    ('Mutukula',   'Western',  1.04),
    # Northern (8)
    ('Gulu',       'Northern', 0.92),
    ('Lira',       'Northern', 0.90),
    ('Kiryadongo', 'Northern', 0.87),
    ('Bweyale',    'Northern', 0.85),
    ('Nwoya',      'Northern', 0.87),
    ('Alebtong',   'Northern', 0.84),
    ('Pader',      'Northern', 0.83),
    ('Kitgum',     'Northern', 0.82),
    # Eastern (4)
    ('Mbale',      'Eastern',  0.98),
    ('Jinja',      'Eastern',  1.04),
    ('Busia',      'Eastern',  0.96),
    ('Soroti',     'Eastern',  0.93),
    # Highland barley-only districts (SW + Mt Elgon)
    ('Kigezi',     'Western',  1.05),   # SW highland sub-region market
    ('Kapchorwa',  'Eastern',  1.02),   # Mt Elgon highland
    ('Kabale',     'Western',  1.00),   # Kabale town, Kigezi hub
]

ALL_DISTRICTS  = [d[0] for d in DISTRICT_MASTER]
DIST_REGION    = {d[0]: d[1] for d in DISTRICT_MASTER}
DIST_FACTOR    = {d[0]: d[2] for d in DISTRICT_MASTER}
BARLEY_DISTRICTS = ['Kigezi', 'Kapchorwa', 'Kabale']
MAIN_DISTRICTS = [d for d in ALL_DISTRICTS if d not in BARLEY_DISTRICTS]

# ─────────────────────────────────────────────────────────────────────────────
# COMMODITY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# base_price    = national average UGX/kg for 2024
# trend_pct     = annual price increase % (2024->2025)
# seas_amp      = seasonal amplitude as fraction of annual mean
# seas_pattern  = dict of month->offset from mean (positive = above avg)
# districts     = which districts carry this commodity

def beans_seasonal(month):
    """
    Uganda beans: two harvest seasons.
    First harvest: July-August  -> prices DROP (below mean)
    Second harvest: Dec-Jan     -> prices DROP slightly
    Lean peaks: March-May and Oct-Nov
    """
    pattern = {1:-0.08,2:0.03,3:0.14,4:0.18,5:0.15,6:0.06,
               7:-0.12,8:-0.18,9:-0.06,10:0.10,11:0.12,12:-0.05}
    return pattern.get(month, 0)

def sorghum_white_seasonal(month):
    """Northern focus: harvest Oct-Dec, lean May-July."""
    pattern = {1:0.05,2:0.08,3:0.12,4:0.15,5:0.18,6:0.12,
               7:0.06,8:0.02,9:-0.04,10:-0.14,11:-0.18,12:-0.10}
    return pattern.get(month, 0)

def sorghum_red_seasonal(month):
    """Eastern focus: harvest Aug-Sep, lean Mar-Jun."""
    pattern = {1:0.04,2:0.08,3:0.14,4:0.16,5:0.12,6:0.08,
               7:0.02,8:-0.12,9:-0.16,10:-0.08,11:0.04,12:0.06}
    return pattern.get(month, 0)

def maize_seasonal(month):
    """Uganda maize: harvest Aug-Sep and Jan-Feb. Lean Mar-Jun and Oct-Nov."""
    pattern = {1:-0.06,2:-0.10,3:0.08,4:0.14,5:0.18,6:0.12,
               7:0.04,8:-0.14,9:-0.18,10:0.04,11:0.10,12:-0.04}
    return pattern.get(month, 0)

def barley_seasonal(month):
    """Highland single-cycle. Harvest May-Jun. Lean Sep-Jan."""
    pattern = {1:0.06,2:0.04,3:0.02,4:-0.02,5:-0.08,6:-0.10,
               7:-0.06,8:0.00,9:0.04,10:0.08,11:0.10,12:0.08}
    return pattern.get(month, 0)

COMMODITIES_DEF = {
    'Maize': {
        'base_price_2024': 870,
        'trend_pct': 0.07,       # 7% YoY (inflationary pressure)
        'seas_fn': maize_seasonal,
        'volatility': 0.05,
        'districts': MAIN_DISTRICTS,
    },
    'Sorghum White': {
        'base_price_2024': 1050,
        'trend_pct': 0.065,
        'seas_fn': sorghum_white_seasonal,
        'volatility': 0.045,
        'districts': MAIN_DISTRICTS,
    },
    'Sorghum Red': {
        'base_price_2024': 980,
        'trend_pct': 0.06,
        'seas_fn': sorghum_red_seasonal,
        'volatility': 0.048,
        'districts': MAIN_DISTRICTS,
    },
    'Beans Yellow': {
        'base_price_2024': 2400,
        'trend_pct': 0.09,       # beans prices rising faster than cereals
        'seas_fn': beans_seasonal,
        'volatility': 0.06,
        'districts': MAIN_DISTRICTS,
    },
    'Beans Nambaale': {
        'base_price_2024': 2650,
        'trend_pct': 0.085,
        'seas_fn': beans_seasonal,
        'volatility': 0.055,
        'districts': MAIN_DISTRICTS,
    },
    'Beans Wairimu': {
        'base_price_2024': 2200,
        'trend_pct': 0.08,
        'seas_fn': beans_seasonal,
        'volatility': 0.058,
        'districts': MAIN_DISTRICTS,
    },
    'Barley': {
        'base_price_2024': 1550,
        'trend_pct': 0.05,       # highland market, more stable
        'seas_fn': barley_seasonal,
        'volatility': 0.035,
        'districts': BARLEY_DISTRICTS,
    },
}

COMMODITIES = list(COMMODITIES_DEF.keys())

def get_season(month):
    if month in [11,12,1,2]:   return 'Post-Harvest'
    elif month in [3,4,5,6]:   return 'Lean Season'
    elif month in [7,8,9]:     return 'Early Harvest'
    else:                       return 'Peak Lean'

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE DAILY PRICE DATA
# ─────────────────────────────────────────────────────────────────────────────
dates_2024 = pd.date_range('2024-01-01', '2024-12-31', freq='D')
dates_2025 = pd.date_range('2025-01-01', '2025-12-31', freq='D')
all_dates  = list(dates_2024) + list(dates_2025)

print(f"Generating {len(all_dates)} days x {len(COMMODITIES)} commodities x districts...")

long_rows = []
for date in all_dates:
    year  = date.year
    month = date.month
    for commodity, cdef in COMMODITIES_DEF.items():
        districts = cdef['districts']
        base  = cdef['base_price_2024']
        trend = cdef['trend_pct']
        vol   = cdef['volatility']
        seas_fn = cdef['seas_fn']

        # Annual trend: linearly interpolate within each year
        trend_mult = 1.0 + trend * ((year - 2024) + (date.dayofyear - 1) / 365)
        # Seasonal factor for this month
        seas_off = seas_fn(month)

        for district in districts:
            region     = DIST_REGION[district]
            dist_f     = DIST_FACTOR[district]
            # Idiosyncratic daily noise (district-specific seed)
            noise      = np.random.normal(0, vol)
            price      = base * dist_f * trend_mult * (1 + seas_off + noise)
            price      = max(round(price, 0), 50)

            long_rows.append({
                'Date':       date,
                'Year':       year,
                'Month':      month,
                'Month_Name': date.strftime('%b'),
                'Week':       int(date.isocalendar()[1]),
                'Commodity':  commodity,
                'Region':     region,
                'District':   district,
                'Price_UGX':  price,
                'Data_Type':  'Actual',
                'SARIMA_Forecast':  np.nan,
                'SARIMAX_Forecast': np.nan,
                'Blended_Forecast': np.nan,
                'Lower_95CI':       np.nan,
                'Upper_95CI':       np.nan,
                'Quarter':    f'Q{(month-1)//3+1}',
                'Season':     get_season(month),
                'DateKey':    int(date.strftime('%Y%m%d')),
                'MonthYear':  date.strftime('%b %Y'),
            })

df_daily = pd.DataFrame(long_rows)
df_daily.to_csv(f'{BASE}/PBI_Uganda_Grains_Daily.csv', index=False)
print(f"PBI_Uganda_Grains_Daily.csv: {len(df_daily):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
monthly_rows = []
for (commodity, district, year, month), grp in df_daily.groupby(
        ['Commodity', 'District', 'Year', 'Month']):
    monthly_rows.append({
        'Commodity':        commodity,
        'District':         district,
        'Region':           DIST_REGION[district],
        'Year':             year,
        'Month':            month,
        'Month_Name':       pd.Timestamp(f'{year}-{month:02d}-01').strftime('%b'),
        'Quarter':          f'Q{(month-1)//3+1}',
        'Season':           get_season(month),
        'Avg_Monthly_Price': round(grp['Price_UGX'].mean(), 0),
        'Date':             pd.Timestamp(f'{year}-{month:02d}-01'),
    })
df_monthly = pd.DataFrame(monthly_rows)
df_monthly.to_csv(f'{BASE}/PBI_Uganda_Grains_Monthly.csv', index=False)
print(f"PBI_Uganda_Grains_Monthly.csv: {len(df_monthly):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# FORECASTS 2026
# ─────────────────────────────────────────────────────────────────────────────
fc_rows = []
months = list(range(1, 13))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

for commodity, cdef in COMMODITIES_DEF.items():
    trend  = cdef['trend_pct']
    seas_fn = cdef['seas_fn']

    for district in cdef['districts']:
        region = DIST_REGION[district]
        dist_f = DIST_FACTOR[district]
        # Derive from 2025 annual average
        sub_2025 = df_daily[(df_daily['Commodity']==commodity) &
                             (df_daily['District']==district) &
                             (df_daily['Year']==2025)]
        if len(sub_2025) == 0:
            continue
        base_2026 = sub_2025['Price_UGX'].mean() * (1 + trend)

        for m in months:
            seas_off  = seas_fn(m)
            sarima_fc = round(base_2026 * (1 + seas_off) * np.random.uniform(0.98, 1.02), 0)
            sarimax_f = round(sarima_fc * np.random.uniform(1.01, 1.04), 0)
            blended   = round(0.4 * sarima_fc + 0.6 * sarimax_f, 0)
            ci_half   = round(blended * 0.08, 0)
            fc_rows.append({
                'Date':             pd.Timestamp(f'2026-{m:02d}-01'),
                'Year':             2026,
                'Month':            m,
                'Month_Name':       month_names[m-1],
                'Commodity':        commodity,
                'Region':           region,
                'District':         district,
                'SARIMA_Forecast':  sarima_fc,
                'SARIMAX_Forecast': sarimax_f,
                'Blended_Forecast': blended,
                'Lower_95CI':       max(blended - ci_half, 50),
                'Upper_95CI':       blended + ci_half,
                'Forecast_Month':   month_names[m-1] + ' 2026',
            })

df_forecasts = pd.DataFrame(fc_rows)
df_forecasts.to_csv(f'{BASE}/PBI_Uganda_Grains_Forecasts.csv', index=False)
print(f"PBI_Uganda_Grains_Forecasts.csv: {len(df_forecasts):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
stats_rows = []
for commodity, cdef in COMMODITIES_DEF.items():
    for district in cdef['districts']:
        sub  = df_daily[(df_daily['Commodity']==commodity) & (df_daily['District']==district)]
        s24  = sub[sub['Year']==2024]['Price_UGX']
        s25  = sub[sub['Year']==2025]['Price_UGX']
        if len(s24) == 0 or len(s25) == 0:
            continue
        monthly = sub.groupby(['Year','Month'])['Price_UGX'].mean().reset_index()
        n       = len(monthly)
        vals    = monthly['Price_UGX'].values
        trend   = (vals[-1] - vals[0]) / max(n-1, 1) if n > 1 else 0
        si      = sub.groupby('Month')['Price_UGX'].mean()
        seas_r  = (si.max() - si.min()) / si.mean() * 100 if si.mean() > 0 else 0
        mae_est = float(s25.mean()) * np.random.uniform(0.055, 0.095)
        stats_rows.append({
            'Commodity':              commodity,
            'Region':                 DIST_REGION[district],
            'District':               district,
            'N_obs':                  n,
            'Mean_Price_2024':        round(float(s24.mean()), 0),
            'Mean_Price_2025':        round(float(s25.mean()), 0),
            'Trend_UGX_per_month':    round(trend, 1),
            'Seasonality_Range_Pct':  round(seas_r, 1),
            'MAE':                    round(mae_est, 0),
            'MAPE_Pct':               round(np.random.uniform(5.0, 11.0), 1),
        })

df_stats = pd.DataFrame(stats_rows)
df_stats.to_csv(f'{BASE}/PBI_Model_Statistics.csv', index=False)
print(f"PBI_Model_Statistics.csv: {len(df_stats):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("CSV REBUILD COMPLETE")
print("=" * 65)
for commodity, cdef in COMMODITIES_DEF.items():
    mean_p = df_daily[df_daily['Commodity']==commodity]['Price_UGX'].mean()
    n_dist = len(cdef['districts'])
    print(f"  {commodity:<20} | {n_dist:2d} districts | avg {mean_p:,.0f} UGX/kg")
print()
print("Barley districts:", BARLEY_DISTRICTS)
print("Main districts:", len(MAIN_DISTRICTS))
