import json
import textwrap

NB = 'Uganda_Grain_SARIMA_Analysis.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
modified = 0

# Helper: wrap a code cell body with a guard condition

def wrap_with_bt_guard(src: str) -> str:
    guard = (
        "# Guard: skip this cell if backtest_df is missing or empty\n"
        "import pandas as pd\n"
        "_BT_EMPTY = ('backtest_df' not in globals()) or (isinstance(globals().get('backtest_df'), pd.DataFrame) and globals().get('backtest_df').empty)\n"
        "if _BT_EMPTY:\n"
        "    print('Backtest results empty or missing; skipping this cell.')\n"
        "else:\n"
    )
    body = textwrap.indent(src, '    ')
    return guard + body

for c in cells:
    if c.get('cell_type') != 'code':
        continue
    src_list = c.get('source', [])
    src = ''.join(src_list)
    new_src = src

    # Section 8 backtesting cell: add safe defaults and required imports
    if 'SECTION 8 - COMPREHENSIVE WALK-FORWARD BACKTESTING' in src:
        pre = (
            "# Section 8 guards and imports\n"
            "import numpy as np, pandas as pd\n"
            "try:\n"
            "    from scipy.stats import shapiro, chi2\n"
            "except Exception:\n"
            "    from scipy.stats import shapiro\n"
            "    from scipy.stats import chi2\n"
            "# Ensure sarima_results exists as dict to allow empty backtest\n"
            "if 'sarima_results' not in globals() or not isinstance(sarima_results, dict):\n"
            "    sarima_results = {}\n"
        )
        if 'Section 8 guards and imports' not in src:
            new_src = pre + src

    # Heatmap and any cell that directly builds plots/tables from backtest_df
    triggers = [
        'pivot = backtest_df.pivot_table',
        'MODEL PERFORMANCE SUMMARY',
        'all_districts_bt = backtest_df',
        'backtest_df.groupby',
        'backtest_df[["Commodity","District"',
    ]
    if any(t in src for t in triggers):
        # Avoid double-wrapping
        if 'Backtest results empty or missing; skipping this cell.' not in src:
            new_src = wrap_with_bt_guard(src)

    if new_src != src:
        c['source'] = [new_src]
        modified += 1

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Patched Section 8+ cells. Cells modified: {modified}')
