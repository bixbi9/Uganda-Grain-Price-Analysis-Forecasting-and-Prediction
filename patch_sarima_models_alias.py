import json

NB = 'Uganda_Grain_SARIMA_Analysis.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
mods = 0

for c in cells:
    if c.get('cell_type') != 'code':
        continue
    src_list = c.get('source', [])
    src = ''.join(src_list)
    if "sarima_models[commodity]['Kampala']" in src:
        lines = src.splitlines()
        new_lines = []
        inserted_preamble = False
        for i, line in enumerate(lines):
            # Insert a preamble near the top once
            if not inserted_preamble and i == 0:
                pre = (
                    "# Guard: define sarima_models alias from sarima_results if missing\n"
                    "if 'sarima_models' not in globals():\n"
                    "    sarima_models = {}\n"
                    "    try:\n"
                    "        if 'sarima_results' in globals() and isinstance(sarima_results, dict):\n"
                    "            sarima_models = {c: {d: sarima_results[c][d].get('model', sarima_results[c][d]) for d in sarima_results.get(c, {})} for c in sarima_results}\n"
                    "    except Exception:\n"
                    "        sarima_models = {}\n"
                )
                new_lines.append(pre)
                inserted_preamble = True
            # Replace exact usage with robust district selection
            if "sarima_models[commodity]['Kampala']" in line:
                indent = line[:len(line) - len(line.lstrip(' '))]
                repl = [
                    f"{indent}models_c = sarima_models.get(commodity, {{}})",
                    f"{indent}dist_rep = 'Kampala' if 'Kampala' in models_c else (next(iter(models_c)) if models_c else None)",
                    f"{indent}if dist_rep is None:",
                    f"{indent}    # No model available; skip this commodity",
                    f"{indent}    continue",
                    f"{indent}model = models_c[dist_rep]",
                ]
                new_lines.extend(repl)
            else:
                new_lines.append(line)
        c['source'] = ['\n'.join(new_lines)]
        mods += 1

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Patched cells with sarima_models alias. Cells modified: {mods}')
