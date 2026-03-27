import json
import re

NB = 'Uganda_Grain_SARIMA_Analysis.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
modified = 0

for c in cells:
    if c.get('cell_type') != 'code':
        continue
    src = ''.join(c.get('source', []))
    orig = src

    # 1) Add pandas import to guard preamble in figure cells
    if 'Guards for standalone execution' in src and 'import pandas as pd' not in src:
        src = src.replace('import os, numpy as np\n', 'import os, numpy as np\nimport pandas as pd\n')

    # 2) Clip errorbar yerr to avoid negative values
    if 'ax_right.errorbar' in src and 'fc_nat - lo_nat' in src and 'yerr_low' not in src:
        lines = src.splitlines()
        new_lines = []
        inserted = False
        for line in lines:
            if (not inserted) and ('ax_right.errorbar' in line):
                indent = line[:len(line) - len(line.lstrip(' '))]
                new_lines.append(f"{indent}yerr_low = np.clip(fc_nat - lo_nat, 0, None)")
                new_lines.append(f"{indent}yerr_high = np.clip(hi_nat - fc_nat, 0, None)")
                inserted = True
            if 'yerr=[fc_nat - lo_nat, hi_nat - fc_nat]' in line:
                line = line.replace('yerr=[fc_nat - lo_nat, hi_nat - fc_nat]', 'yerr=[yerr_low, yerr_high]')
            new_lines.append(line)
        src = '\n'.join(new_lines)
    elif 'ax_right.errorbar' in src and 'yerr=[fc_nat - lo_nat, hi_nat - fc_nat]' in src:
        src = src.replace('yerr=[fc_nat - lo_nat, hi_nat - fc_nat]', 'yerr=[yerr_low, yerr_high]')

    # 3) Guard KEY_DISTRICTS / BARLEY_DISTRICTS in LSTM cell
    if 'SECTION 11C' in src and 'LSTM (Long Short-Term Memory)' in src and 'Guard: ensure KEY_DISTRICTS' not in src:
        lines = src.splitlines()
        new_lines = []
        inserted = False
        for line in lines:
            if (not inserted) and line.strip().startswith('WINDOW = 12'):
                guard_lines = [
                    '# Guard: ensure KEY_DISTRICTS / BARLEY_DISTRICTS exist',
                    'import pandas as pd',
                    "if 'KEY_DISTRICTS' not in globals():",
                    "    if 'COMMODITY_DISTRICTS' in globals() and COMMODITY_DISTRICTS:",
                    "        _all_d = {d for ds in COMMODITY_DISTRICTS.values() for d in ds}",
                    "        KEY_DISTRICTS = sorted(_all_d)[:8]",
                    "    elif 'df_monthly' in globals():",
                    "        try:",
                    "            KEY_DISTRICTS = sorted(pd.Series(df_monthly.get('District', [])).dropna().unique().tolist())[:8]",
                    "        except Exception:",
                    "            KEY_DISTRICTS = []",
                    "    else:",
                    "        KEY_DISTRICTS = []",
                    "if 'BARLEY_DISTRICTS' not in globals():",
                    "    if 'COMMODITY_DISTRICTS' in globals():",
                    "        BARLEY_DISTRICTS = COMMODITY_DISTRICTS.get('Barley', [])",
                    "    else:",
                    "        BARLEY_DISTRICTS = []",
                ]
                new_lines.extend(guard_lines)
                inserted = True
            new_lines.append(line)
        src = '\n'.join(new_lines)

    # 4) Guard ml_datasets / COMMODITIES in Section 11D forecasts cell
    if 'SECTION 11D' in src and '12-MONTH FORECASTS FROM RF' in src and 'Guard: ensure ml_datasets' not in src:
        lines = src.splitlines()
        new_lines = []
        inserted = False
        for line in lines:
            if (not inserted) and line.strip().startswith('fc_dates'):
                guard_lines = [
                    '# Guard: ensure ml_datasets/COMMODITIES exist',
                    "if 'ml_datasets' not in globals():",
                    "    ml_datasets = {}",
                    "if 'COMMODITIES' not in globals():",
                    "    if ml_datasets:",
                    "        COMMODITIES = sorted(ml_datasets.keys())",
                    "    elif 'df_monthly' in globals():",
                    "        try:",
                    "            COMMODITIES = sorted(pd.Series(df_monthly.get('Commodity', [])).dropna().unique().tolist())",
                    "        except Exception:",
                    "            COMMODITIES = []",
                    "    else:",
                    "        COMMODITIES = []",
                ]
                # Ensure pandas is imported if not already
                if 'import pandas as pd' not in src:
                    guard_lines.insert(1, 'import pandas as pd')
                new_lines.extend(guard_lines)
                inserted = True
            new_lines.append(line)
        src = '\n'.join(new_lines)

    # 5) Section 10 forecast table: avoid KeyError for missing districts
    if 'SECTION 10' in src and 'FINAL FORECAST TABLE' in src:
        lines = src.splitlines()
        new_lines = []
        inserted_table_guard = False
        in_seasonal = False
        for line in lines:
            if 'SEASONAL OUTLOOK' in line:
                in_seasonal = True

            # Replace DISTRICTS usage with districts
            if 'for district in DISTRICTS' in line:
                line = line.replace('for district in DISTRICTS', 'for district in districts')
            if 'for d in DISTRICTS' in line:
                line = line.replace('for d in DISTRICTS', 'for d in districts')

            new_lines.append(line)

            # Insert district guard for table after header divider
            if (not in_seasonal) and (not inserted_table_guard) and ('"  " + "â”€" * 86' in line or '"  " + "-" * 86' in line):
                indent = line[:len(line) - len(line.lstrip(' '))]
                guard_block = [
                    f"{indent}districts = [d for d in DISTRICTS if d in sarima_forecasts.get(commodity, {{}}) and d in sarimax_forecasts.get(commodity, {{}})]",
                    f"{indent}if not districts:",
                    f"{indent}    districts = sorted(set(sarima_forecasts.get(commodity, {{}})) & set(sarimax_forecasts.get(commodity, {{}})))",
                    f"{indent}if not districts:",
                    f"{indent}    print('  (No forecast data available for this commodity)')",
                    f"{indent}    continue",
                ]
                new_lines.extend(guard_block)
                inserted_table_guard = True

            # Insert district guard for seasonal loop right after its for-commodity line
            if in_seasonal and line.strip() == 'for commodity in COMMODITIES:':
                indent = line[:len(line) - len(line.lstrip(' '))] + '    '
                guard_block = [
                    f"{indent}districts = [d for d in DISTRICTS if d in sarima_forecasts.get(commodity, {{}}) and d in sarimax_forecasts.get(commodity, {{}})]",
                    f"{indent}if not districts:",
                    f"{indent}    districts = sorted(set(sarima_forecasts.get(commodity, {{}})) & set(sarimax_forecasts.get(commodity, {{}})))",
                    f"{indent}if not districts:",
                    f"{indent}    print(f'\\n  {{commodity}}  (no forecast data available)')",
                    f"{indent}    continue",
                ]
                new_lines.extend(guard_block)

        src = '\n'.join(new_lines)

    if src != orig:
        c['source'] = [src]
        modified += 1

if modified:
    with open(NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Patched notebook cells: {modified}')
