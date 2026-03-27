import sys, json
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

with open('Uganda_Grain_SARIMA_Analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][4]['source'])

# Remove parse_dates arguments from all three read_csv calls
src = src.replace(
    "df_daily    = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Daily.csv',    parse_dates=['Date'])",
    "df_daily    = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Daily.csv')"
)
src = src.replace(
    "df_monthly  = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Monthly.csv',  parse_dates=['Date'])",
    "df_monthly  = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Monthly.csv')"
)
src = src.replace(
    "df_forecast = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Forecasts.csv',parse_dates=['Date'])",
    "df_forecast = pd.read_csv(f'{BASE}/PBI_Uganda_Grains_Forecasts.csv')"
)

# Insert explicit datetime conversion block after df_stats load
src = src.replace(
    "df_stats    = pd.read_csv(f'{BASE}/PBI_Model_Statistics.csv')",
    (
        "df_stats    = pd.read_csv(f'{BASE}/PBI_Model_Statistics.csv')\n"
        "\n"
        "# Explicitly convert Date columns to datetime after loading\n"
        "for _df in [df_daily, df_monthly, df_forecast]:\n"
        "    if 'Date' in _df.columns:\n"
        "        _df['Date'] = pd.to_datetime(_df['Date'], dayfirst=False)\n"
        "# Also convert MonthYear if present (used for groupby/pivots)\n"
        "for _df in [df_daily]:\n"
        "    if 'MonthYear' in _df.columns and _df['MonthYear'].dtype == object:\n"
        "        pass  # keep as string label for groupby titles"
    )
)

nb['cells'][4]['source'] = [src]
with open('Uganda_Grain_SARIMA_Analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Cell 4 date parsing fixed.")
# Verify
count = src.count("pd.to_datetime")
print(f"  pd.to_datetime calls: {count}")
print(f"  parse_dates remaining: {src.count('parse_dates')}")
