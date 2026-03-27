import json
import re

NB = 'Uganda_Grain_SARIMA_Analysis.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
mods = 0

for c in cells:
    if c.get('cell_type') != 'code':
        continue
    src = ''.join(c.get('source', []))
    orig = src

    # Fix malformed multiplier '_0.1' -> '* 0.1'
    src = re.sub(r"\)\s*_\s*0\.1", ") * 0.1", src)

    # Identify errorbar usage with fc_nat/lo_nat/hi_nat
    if 'errorbar' in src and 'yerr=' in src and all(k in src for k in ['fc_nat', 'lo_nat', 'hi_nat']):
        lines = src.splitlines()
        new_lines = []
        inserted_clip = False
        for i, line in enumerate(lines):
            # Insert clipping definitions once before calling errorbar
            if (not inserted_clip) and ('errorbar' in line) and ('yerr=' in line):
                # Add computations above current line
                new_lines.append("yerr_low  = np.clip(fc_nat - lo_nat, 0, None)")
                new_lines.append("yerr_high = np.clip(hi_nat - fc_nat, 0, None)")
                inserted_clip = True
                # Replace yerr argument in this line
                line = re.sub(r"yerr\s*=\s*\[[^\]]*\]", "yerr=[yerr_low, yerr_high]", line)
            new_lines.append(line)
        src = '\n'.join(new_lines)

    if src != orig:
        c['source'] = [src]
        mods += 1

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Patched errorbar yerr and typos. Cells modified: {mods}')
