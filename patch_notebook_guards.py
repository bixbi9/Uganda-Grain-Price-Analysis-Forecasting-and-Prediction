import json
import re

NB = 'Uganda_Grain_SARIMA_Analysis.ipynb'

preamble = (
    "# Guards for standalone execution\n"
    "import os, numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mticker\n"
    "globals().setdefault('WINDOW', 12)\n"
    "# Ensure expected dicts exist even if ML sections were not run\n"
    "for _name in ['SVR_RESULTS','RF_RESULTS','LSTM_RESULTS','ml_datasets','ml_forecasts','blended_forecasts']:\n"
    "    if _name not in globals():\n"
    "        globals()[_name] = {}\n"
    "# Recover commodity lists and mapping if needed\n"
    "if 'COMMODITIES' not in globals():\n"
    "    if 'COMMODITY_DISTRICTS' in globals():\n"
    "        COMMODITIES = list(COMMODITY_DISTRICTS.keys())\n"
    "    elif 'df_monthly' in globals():\n"
    "        try:\n"
    "            COMMODITIES = sorted(pd.Series(df_monthly.get('Commodity', [])).dropna().unique().tolist())\n"
    "        except Exception:\n"
    "            COMMODITIES = []\n"
    "    else:\n"
    "        COMMODITIES = []\n"
    "if 'COMMODITY_DISTRICTS' not in globals():\n"
    "    if 'df_monthly' in globals():\n"
    "        try:\n"
    "            COMMODITY_DISTRICTS = {c: sorted(df_monthly[df_monthly['Commodity']==c]['District'].dropna().unique().tolist()) for c in COMMODITIES}\n"
    "        except Exception:\n"
    "            COMMODITY_DISTRICTS = {c: [] for c in COMMODITIES}\n"
    "    else:\n"
    "        COMMODITY_DISTRICTS = {c: [] for c in COMMODITIES}\n"
    "if 'PALETTE' not in globals():\n"
    "    PALETTE = {}\n"
)

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
changed = 0

fig_headers = [
    '# FIGURE 15: ML MODEL IN-SAMPLE FIT',
    '# FIGURE 16: SEASONAL PATTERN COMPARISON',
    '# FIGURE 17: DISTRICT PRICE SPREAD',
]

for i, c in enumerate(cells):
    if c.get('cell_type') != 'code':
        continue
    src_list = c.get('source', [])
    src = ''.join(src_list)

    # Add guards to figure cells
    if any(h in src for h in fig_headers):
        if 'Guards for standalone execution' not in src:
            new_src = preamble + "\n" + src
            c['source'] = [new_src]
            changed += 1

    # Fix forecast unpacking: ensure fc is at least 1D before indexing
    if 'y_hat = fc[0]' in src and 'np.atleast_1d' not in src:
        # Try to preserve indentation by finding the line and re-indenting
        lines = src.splitlines()
        for li, line in enumerate(lines):
            if line.strip().startswith('y_hat = fc[0]'):
                indent = line[:len(line) - len(line.lstrip(' '))]
                lines[li] = f"{indent}fc = np.atleast_1d(fc)\n{indent}y_hat = fc[0]"
                changed += 1
                break
        c['source'] = ['\n'.join(lines)]

# Save only if changes made
if changed:
    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched notebook. Cells modified: {changed}")
