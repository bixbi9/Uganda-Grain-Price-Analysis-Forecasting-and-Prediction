"""
Fix all cells that unpack model.forecast() as a tuple.
forecast() returns a dict: {'forecast': arr, 'lower_95': arr, 'upper_95': arr}
"""
import sys, json
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

with open('Uganda_Grain_SARIMA_Analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 19: SARIMAX fitting
src19 = ''.join(nb['cells'][19]['source'])
src19 = src19.replace(
    'base_fc, lo95, hi95 = model.forecast(12)',
    '_fc19 = model.forecast(12)\n'
    '        base_fc = _fc19[\'forecast\']\n'
    '        lo95    = _fc19[\'lower_95\']\n'
    '        hi95    = _fc19[\'upper_95\']'
)
nb['cells'][19]['source'] = [src19]
print('Cell 19 fixed')

# Cell 20: Build forecast dicts
src20 = ''.join(nb['cells'][20]['source'])
src20 = src20.replace(
    'fc, lo95, hi95 = model.forecast(12)',
    '_fc20 = model.forecast(12)\n'
    '        fc   = _fc20[\'forecast\']\n'
    '        lo95 = _fc20[\'lower_95\']\n'
    '        hi95 = _fc20[\'upper_95\']'
)
nb['cells'][20]['source'] = [src20]
print('Cell 20 fixed')

# Cell 27: Walk-forward backtest  (fc, _, _ = m.forecast(1))
src27 = ''.join(nb['cells'][27]['source'])
src27 = src27.replace(
    'fc, _, _ = m.forecast(1)',
    '_fc27 = m.forecast(1)\n'
    '            fc = _fc27[\'forecast\']'
)
nb['cells'][27]['source'] = [src27]
print('Cell 27 fixed')

with open('Uganda_Grain_SARIMA_Analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('All forecast() unpack calls fixed.')
