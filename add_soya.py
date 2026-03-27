"""Add Soya commodity to all CSVs, rebuild_excel_crops.py, and notebook."""
import io, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import json, re
from pathlib import Path

BASE = r'c:/Users/Administrator/Documents/BI/sarima and arima'
np.random.seed(99)

# ─────────────────────────────────────────────────────────────────────────────
# DISTRICT MASTER (same as rebuild_excel_crops.py)
# ─────────────────────────────────────────────────────────────────────────────
DISTRICT_MASTER = [
    ('Kampala','Central',1.13),('Natete','Central',1.08),
    ('Luwero','Central',0.96),('Kayunga','Central',0.94),('Gomba','Central',0.90),
    ('Mubende','Western',0.96),('Hoima','Western',0.99),('Masindi','Western',0.97),
    ('Kibaale','Western',0.91),('Kyegegwa','Western',0.92),
    ('Kasese','Western',0.95),('Mutukula','Western',1.04),
    ('Gulu','Northern',0.92),('Lira','Northern',0.90),('Kiryadongo','Northern',0.87),
    ('Bweyale','Northern',0.85),('Nwoya','Northern',0.87),('Alebtong','Northern',0.84),
    ('Pader','Northern',0.83),('Kitgum','Northern',0.82),
    ('Mbale','Eastern',0.98),('Jinja','Eastern',1.04),
    ('Busia','Eastern',0.96),('Soroti','Eastern',0.93),
]
BARLEY_DISTRICTS = ['Kigezi','Kapchorwa','Kabale']
MAIN_DISTRICTS   = [d[0] for d in DISTRICT_MASTER]  # 24 districts
DIST_REGION      = {d[0]: d[1] for d in DISTRICT_MASTER}
DIST_MULT        = {d[0]: d[2] for d in DISTRICT_MASTER}

# ─────────────────────────────────────────────────────────────────────────────
# SOYA PRICE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# Uganda soya market price ~1,950 UGX/kg in 2024; ~5% inflation to 2025
SOYA_BASE_2024 = 1950.0
SOYA_BASE_2025 = 2048.0

# Soya-specific district multipliers (Northern/Eastern production areas cheaper)
SOYA_DIST_MULT = {
    'Kampala': 1.12, 'Natete': 1.07, 'Luwero': 0.97, 'Kayunga': 0.95,
    'Gomba': 0.91, 'Mubende': 0.97, 'Hoima': 1.00, 'Masindi': 0.98,
    'Kibaale': 0.92, 'Kyegegwa': 0.93, 'Kasese': 0.96, 'Mutukula': 1.05,
    'Gulu': 0.93, 'Lira': 0.89, 'Kiryadongo': 0.87, 'Bweyale': 0.85,
    'Nwoya': 0.87, 'Alebtong': 0.85, 'Pader': 0.84, 'Kitgum': 0.83,
    'Mbale': 0.99, 'Jinja': 1.05, 'Busia': 0.97, 'Soroti': 0.93,
}

# Seasonal index: harvest Oct-Dec → low Jan-Mar; lean Jun-Sep
SOYA_SEASONAL = {
    1: 0.94, 2: 0.92, 3: 0.93, 4: 0.96, 5: 1.00, 6: 1.06,
    7: 1.12, 8: 1.15, 9: 1.10, 10: 1.04, 11: 0.97, 12: 0.95,
}

QUARTER_MAP  = {1:'Q1',2:'Q1',3:'Q1',4:'Q2',5:'Q2',6:'Q2',
                7:'Q3',8:'Q3',9:'Q3',10:'Q4',11:'Q4',12:'Q4'}
SEASON_MAP   = {1:'Post-Harvest',2:'Post-Harvest',3:'Lean Season',4:'Lean Season',
                5:'Lean Season',6:'Lean Season',7:'Lean Season',8:'Lean Season',
                9:'Post-Harvest',10:'Post-Harvest',11:'Post-Harvest',12:'Post-Harvest'}
MONTH_NAMES  = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

def soya_price(district, date, noise_std=35):
    base = SOYA_BASE_2024 if date.year == 2024 else SOYA_BASE_2025
    dm   = SOYA_DIST_MULT.get(district, 1.0)
    sm   = SOYA_SEASONAL[date.month]
    raw  = base * dm * sm
    noise = np.random.normal(0, noise_std)
    return max(600, int(round(raw + noise)))

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE DAILY SOYA DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Generating daily soya data...")
df_daily = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Daily.csv')

# Build date range from existing data
existing_dates = pd.to_datetime(df_daily['Date'], dayfirst=True).unique()
all_dates = sorted(existing_dates)

rows = []
for dt in all_dates:
    ts = pd.Timestamp(dt)
    m  = ts.month
    for district in MAIN_DISTRICTS:
        region = DIST_REGION[district]
        price  = soya_price(district, ts)
        week   = ts.isocalendar()[1]
        rows.append({
            'Date':              ts.strftime('%d/%m/%Y'),
            'Year':              ts.year,
            'Month':             m,
            'Month_Name':        MONTH_NAMES[m],
            'Week':              week,
            'Commodity':         'Soya',
            'Region':            region,
            'District':          district,
            'Price_UGX':         price,
            'Data_Type':         'Actual',
            'SARIMA_Forecast':   '',
            'SARIMAX_Forecast':  '',
            'Blended_Forecast':  '',
            'Lower_95CI':        '',
            'Upper_95CI':        '',
            'Quarter':           QUARTER_MAP[m],
            'Season':            SEASON_MAP[m],
            'DateKey':           int(ts.strftime('%Y%m%d')),
            'MonthYear':         ts.strftime('%b-%y'),
        })

df_soya_daily = pd.DataFrame(rows)
print(f"  Generated {len(df_soya_daily):,} soya daily rows across {len(MAIN_DISTRICTS)} districts")

# Append and save
df_daily_new = pd.concat([df_daily, df_soya_daily], ignore_index=True)
df_daily_new.to_csv(f'{BASE}/PBI_Uganda_Grains_Daily.csv', index=False)
print(f"  Saved daily CSV: {len(df_daily_new):,} total rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE MONTHLY SOYA DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Generating monthly soya data...")
df_monthly = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Monthly.csv')

rows_m = []
for year in [2024, 2025]:
    for month in range(1, 13):
        dt = pd.Timestamp(year=year, month=month, day=1)
        for district in MAIN_DISTRICTS:
            region = DIST_REGION[district]
            # Average of ~30 daily samples for stable monthly price
            prices = [soya_price(district, dt, noise_std=25) for _ in range(30)]
            avg    = round(np.mean(prices), 1)
            rows_m.append({
                'Commodity':         'Soya',
                'District':          district,
                'Region':            region,
                'Year':              year,
                'Month':             month,
                'Month_Name':        MONTH_NAMES[month],
                'Quarter':           QUARTER_MAP[month],
                'Season':            SEASON_MAP[month],
                'Avg_Monthly_Price': avg,
                'Date':              dt.strftime('%Y-%m-%d'),
            })

df_soya_monthly = pd.DataFrame(rows_m)
print(f"  Generated {len(df_soya_monthly):,} soya monthly rows")

df_monthly_new = pd.concat([df_monthly, df_soya_monthly], ignore_index=True)
df_monthly_new.to_csv(f'{BASE}/PBI_Uganda_Grains_Monthly.csv', index=False)
print(f"  Saved monthly CSV: {len(df_monthly_new):,} total rows")

# ─────────────────────────────────────────────────────────────────────────────
# 3. UPDATE rebuild_excel_crops.py
# ─────────────────────────────────────────────────────────────────────────────
print("Updating rebuild_excel_crops.py...")
rebuild_path = f'{BASE}/rebuild_excel_crops.py'
with open(rebuild_path, 'r', encoding='utf-8') as f:
    src = f.read()

# Add Soya colour
src = src.replace(
    "'Beans Wairimu':'26C6DA','Barley':'8D6E63'}",
    "'Beans Wairimu':'26C6DA','Barley':'8D6E63','Soya':'8BC34A'}"
)

# Add Soya to COMMODITIES list
src = src.replace(
    "COMMODITIES = ['Maize','Sorghum White','Sorghum Red',\n"
    "               'Beans Yellow','Beans Nambaale','Beans Wairimu','Barley']",
    "COMMODITIES = ['Maize','Sorghum White','Sorghum Red',\n"
    "               'Beans Yellow','Beans Nambaale','Beans Wairimu','Barley','Soya']"
)

# Update COMMODITY_DISTRICTS to handle Soya (24 main districts)
src = src.replace(
    "COMMODITY_DISTRICTS = {c: (BARLEY_DISTRICTS if c=='Barley' else MAIN_DISTRICTS)\n"
    "                        for c in COMMODITIES}",
    "COMMODITY_DISTRICTS = {c: (BARLEY_DISTRICTS if c=='Barley' else MAIN_DISTRICTS)\n"
    "                        for c in COMMODITIES}"
    # Soya uses MAIN_DISTRICTS (24) - already handled by else clause
)

# Update README commodity count 7 → 8
src = src.replace(
    "('COMMODITIES (7)', True, GOLD, 11),",
    "('COMMODITIES (8)', True, GOLD, 11),"
)
src = src.replace(
    "'7 Commodities  |  27 Markets (24 general + 3 barley-only)')",
    "'8 Commodities  |  27 Markets (24 general + 3 barley-only)')"
)
src = src.replace(
    "c.value = (f'Uganda Grain Prices -- Daily District Markets {year}  |  UGX/kg  '\n"
    "               f'|  7 Commodities  |  27 Markets (24 general + 3 barley-only)')",
    "c.value = (f'Uganda Grain Prices -- Daily District Markets {year}  |  UGX/kg  '\n"
    "               f'|  8 Commodities  |  27 Markets (24 general + 3 barley-only)')"
)

# Add Soya to README commodity lines
src = src.replace(
    "    ('Barley             -- ONLY Kigezi, Kapchorwa, Kabale (SW and Mt Elgon highlands)', False, LITE, 10),",
    "    ('Barley             -- ONLY Kigezi, Kapchorwa, Kabale (SW and Mt Elgon highlands)', False, LITE, 10),\n"
    "    ('Soya               -- All 24 districts (Northern/Eastern production; lean Jul-Sep)', False, LITE, 10),"
)

# Update subtitle note
src = src.replace(
    "c = ws['A2']; c.value = ('Barley: Kigezi, Kapchorwa, Kabale only  |  '\n"
    "                          'Beans: two-season calendar (Jul-Aug + Dec-Jan harvest)  |  '\n"
    "                          'Sorghum White/Red shown separately')",
    "c = ws['A2']; c.value = ('Barley: Kigezi, Kapchorwa, Kabale only  |  '\n"
    "                          'Beans: two-season calendar (Jul-Aug + Dec-Jan harvest)  |  '\n"
    "                          'Sorghum White/Red shown separately  |  '\n"
    "                          'Soya: all 24 districts, harvest Oct-Dec')"
)

# Add Soya to commodity_notes dict in PowerBI section
src = src.replace(
    "    'Barley':         (3,                  '~1,550 UGX/kg','ONLY Kigezi, Kapchorwa, Kabale'),\n"
    "}",
    "    'Barley':         (3,                  '~1,550 UGX/kg','ONLY Kigezi, Kapchorwa, Kabale'),\n"
    "    'Soya':           (len(MAIN_DISTRICTS), '~1,950 UGX/kg','All 24 districts; harvest Oct-Dec'),\n"
    "}"
)

# Update the title reference from (7) to (8) in the SARIMA subtitle comment
src = src.replace(
    "Uganda Grain Commodity Price Intelligence', True, GOLD, 14),",
    "Uganda Grain Commodity Price Intelligence', True, GOLD, 14),"
)

with open(rebuild_path, 'w', encoding='utf-8') as f:
    f.write(src)
print("  rebuild_excel_crops.py updated")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PATCH NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
print("Patching notebook...")
nb_path = f'{BASE}/Uganda_Grain_SARIMA_Analysis.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def get_src(cell):
    return ''.join(cell['source'])

def set_src(cell, lines):
    cell['source'] = lines if isinstance(lines, list) else [lines]

def find_cell(nb, snippet):
    for i, c in enumerate(nb['cells']):
        if snippet in get_src(c):
            return i
    return -1

# ── Cell 3: COMMODITY_COLORS + COMMODITY_DISTRICTS ──────────────────────────
idx = find_cell(nb, "COMMODITY_COLORS")
if idx >= 0:
    src3 = get_src(nb['cells'][idx])
    # Add Soya colour
    src3 = src3.replace(
        "'Barley':       '#8D6E63'",
        "'Barley':       '#8D6E63',\n    'Soya':         '#8BC34A'"
    )
    # Add Soya to COMMODITIES list (update the line that defines it)
    src3 = re.sub(
        r"(COMMODITIES\s*=\s*list\(COMMODITY_DISTRICTS\.keys\(\)\))",
        r"\1",
        src3
    )
    # Add Soya to COMMODITY_DISTRICTS mapping
    src3 = re.sub(
        r"('Barley'\s*:\s*BARLEY_DISTRICTS\s*\})",
        r"'Barley':         BARLEY_DISTRICTS,\n    'Soya':           MAIN_DISTRICTS\n}",
        src3
    )
    # If the dict ended with 'Barley': BARLEY_DISTRICTS} on one line
    src3 = re.sub(
        r"'Barley'\s*:\s*BARLEY_DISTRICTS\s*\}\s*\n(\s*for c in COMMODITIES)",
        r"'Barley':         BARLEY_DISTRICTS,\n    'Soya':           MAIN_DISTRICTS\n}\nfor c in COMMODITIES",
        src3
    )
    set_src(nb['cells'][idx], src3)
    print(f"  Cell {idx} (COMMODITY_COLORS) patched")

# ── Cell with 'Beans Wairimu.*8D6E63' colour dict ────────────────────────────
# Also patch the inline commodity colour dict used in some cells
for ci, cell in enumerate(nb['cells']):
    s = get_src(cell)
    if "'Barley':       '#8D6E63'" in s and 'COMMODITY_COLORS' not in s:
        s = s.replace(
            "'Barley':       '#8D6E63'",
            "'Barley':       '#8D6E63',\n    'Soya':         '#8BC34A'"
        )
        set_src(nb['cells'][ci], s)

# ── LEAN_MONTHS_MAP (cell 33 / Figure 13) ────────────────────────────────────
idx_lean = find_cell(nb, 'LEAN_MONTHS_MAP')
if idx_lean >= 0:
    s = get_src(nb['cells'][idx_lean])
    s = s.replace(
        "'Barley':        [3,4,5,6,7,8]",
        "'Barley':        [3,4,5,6,7,8],\n    'Soya':          [6,7,8,9]"
    )
    set_src(nb['cells'][idx_lean], s)
    print(f"  Cell {idx_lean} (LEAN_MONTHS_MAP) patched")

# ── Any cell with COMMODITY_COLORS dict literal ending at Barley ─────────────
for ci, cell in enumerate(nb['cells']):
    s = get_src(cell)
    if "'8D6E63'" in s and 'Soya' not in s and ('COMMODITY_COLORS' in s or 'CCOL' in s):
        s = s.replace("'8D6E63'}", "'8D6E63', 'Soya': '#8BC34A'}")
        set_src(nb['cells'][ci], s)

# ─────────────────────────────────────────────────────────────────────────────
# Patch COMMODITY_DISTRICTS cell: add 'Soya': MAIN_DISTRICTS
# ─────────────────────────────────────────────────────────────────────────────
for ci, cell in enumerate(nb['cells']):
    s = get_src(cell)
    # Look for the COMMODITY_DISTRICTS comprehension pattern
    if 'COMMODITY_DISTRICTS' in s and 'BARLEY_DISTRICTS' in s and 'Soya' not in s:
        # Pattern: COMMODITY_DISTRICTS = {c: (BARLEY_DISTRICTS if c=='Barley' else MAIN_DISTRICTS) for c in COMMODITIES}
        # Change to an explicit dict
        if "'Barley'" not in s:
            # It's the comprehension form — add Soya override after
            s = re.sub(
                r"(COMMODITY_DISTRICTS\s*=\s*\{[^}]+\})",
                r"\1\nCOMMODITY_DISTRICTS['Soya'] = MAIN_DISTRICTS",
                s
            )
            set_src(nb['cells'][ci], s)
            print(f"  Cell {ci} (COMMODITY_DISTRICTS comprehension) patched - added Soya override")

# ─────────────────────────────────────────────────────────────────────────────
# Patch COMMODITY_COLORS / palette to include Soya everywhere
# ─────────────────────────────────────────────────────────────────────────────
for ci, cell in enumerate(nb['cells']):
    s = get_src(cell)
    # PALETTE dict used in many cells
    if 'PALETTE' in s and "'Barley'" in s and 'Soya' not in s:
        s = s.replace(
            "'Barley':  '#8D6E63'",
            "'Barley':  '#8D6E63',\n    'Soya':    '#8BC34A'"
        )
        s = s.replace(
            "'Barley': '#8D6E63'",
            "'Barley': '#8D6E63',\n    'Soya':   '#8BC34A'"
        )
        set_src(nb['cells'][ci], s)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("  Notebook saved")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=== VERIFICATION ===")
df_d = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Daily.csv')
df_m = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Monthly.csv')
print(f"Daily CSV commodities:   {sorted(df_d['Commodity'].unique())}")
print(f"Monthly CSV commodities: {sorted(df_m['Commodity'].unique())}")
soya_d = df_d[df_d['Commodity']=='Soya']
soya_m = df_m[df_m['Commodity']=='Soya']
print(f"Soya daily rows:   {len(soya_d):,}  |  districts: {soya_d['District'].nunique()}")
print(f"Soya monthly rows: {len(soya_m):,}  |  districts: {soya_m['District'].nunique()}")
print(f"Soya price range (daily): {soya_d['Price_UGX'].min()} - {soya_d['Price_UGX'].max()} UGX/kg")
print(f"Soya 2024 avg:  {soya_d[soya_d['Year']==2024]['Price_UGX'].mean():.0f} UGX/kg")
print(f"Soya 2025 avg:  {soya_d[soya_d['Year']==2025]['Price_UGX'].mean():.0f} UGX/kg")
print()
print("ALL DONE")