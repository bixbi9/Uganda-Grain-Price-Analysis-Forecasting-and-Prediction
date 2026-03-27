"""
fix_all_errors.py — comprehensive notebook repair
Fixes:
  1. Cells 22, 23 : subplots(3,8) -> subplots(n_comm,8)  [7 commodities, not 3]
  2. Cell 27      : y_hat = fc[0] guard (fc is array from dict)
  3. Cells 43-45  : robust rewrites of ML chart cells
"""
import json, sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

NB = 'Uganda_Grain_SARIMA_Analysis.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ===========================================================================
# FIX 1 — Cells 22 & 23: subplots rows must match COMMODITIES count (7)
# ===========================================================================
for idx in [22, 23]:
    src = ''.join(cells[idx]['source'])
    src = src.replace(
        'fig, axes = plt.subplots(3, 8, figsize=(52, 18))',
        'n_comm = len(COMMODITIES)\n'
        'fig, axes = plt.subplots(n_comm, 8, figsize=(52, n_comm * 6))'
    )
    cells[idx]['source'] = [src]
    print(f'Cell {idx}: subplots rows fixed -> n_comm x 8')

# ===========================================================================
# FIX 2 — Cell 27: ensure fc[0] works (fc is numpy array from dict)
# ===========================================================================
src27 = ''.join(cells[27]['source'])
# The fix was already applied: _fc27 = m.forecast(1); fc = _fc27['forecast']
# Verify y_hat = fc[0] is present
if 'y_hat = fc[0]' in src27:
    print('Cell 27: y_hat = fc[0] already correct')
else:
    src27 = src27.replace(
        "fc = _fc27['forecast']",
        "fc = np.atleast_1d(_fc27['forecast'])"
    )
    cells[27]['source'] = [src27]
    print('Cell 27: added atleast_1d guard on fc')

# ===========================================================================
# FIX 3 — Cell 43: Figure 15 — robust ML in-sample fit chart
# ===========================================================================
FIG15 = """\
# ============================================================================
# FIGURE 15: ML MODEL IN-SAMPLE FIT — Actual vs SVR / RF / LSTM
# ============================================================================
# For each commodity, shows how well SVR, Random Forest, and LSTM track the
# historical 2024-2025 monthly prices for a representative district.
#
# LEFT  — Fitted price curves overlaid on actual monthly prices.
#   Solid white  = actual prices
#   Dashed pink  = SVR fit  (kernel regression on lag + seasonal features)
#   Dashed green = Random Forest fit  (ensemble of 300 decision trees)
#   Dash-dot cyan= LSTM fit  (recurrent neural network memory)
#
# RIGHT — Residuals: actual minus predicted (positive = under-prediction).
#   Bar colours match model colours. Annotated with RMSE per model.
# ----------------------------------------------------------------------------

REPR_DISTRICT = {
    'Maize':         'Kampala',
    'Sorghum White': 'Gulu',
    'Sorghum Red':   'Mbale',
    'Beans Yellow':  'Mubende',
    'Beans Nambaale':'Kampala',
    'Beans Wairimu': 'Jinja',
    'Barley':        'Kabale',
}

n_comm = len(COMMODITIES)
fig, axes = plt.subplots(n_comm, 2, figsize=(26, n_comm * 4.2))
fig.patch.set_facecolor('#0A1520')
fig.suptitle('ML Model In-Sample Fit  —  Actual vs SVR / Random Forest / LSTM  (2024-2025)',
             fontsize=14, fontweight='bold', color='white', y=1.002)

for ci, commodity in enumerate(COMMODITIES):
    col  = PALETTE.get(commodity, '#888888')
    dist = REPR_DISTRICT.get(commodity, 'Kampala')
    ax_l = axes[ci, 0]
    ax_r = axes[ci, 1]
    ax_l.set_facecolor('#0F1923')
    ax_r.set_facecolor('#0F1923')

    # Monthly actuals for representative district
    price_col = ('Avg_Monthly_Price'
                 if 'Avg_Monthly_Price' in df_monthly.columns else 'Price_UGX')
    mask = ((df_monthly['Commodity'] == commodity) &
            (df_monthly['District']  == dist))
    sub  = df_monthly[mask].sort_values('Date')

    if len(sub) < 6:
        for ax in [ax_l, ax_r]:
            ax.text(0.5, 0.5, f'Insufficient data\\n{commodity}/{dist}',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#8A9BAB', fontsize=9)
            ax.axis('off')
        continue

    dates_act = pd.to_datetime(sub['Date']).values
    y_act     = sub[price_col].values.astype(float)

    # Plot actuals
    ax_l.plot(dates_act, y_act, color='#FFFFFF', linewidth=2.4,
              label='Actual', zorder=6)

    residuals = {}

    # --- SVR in-sample fit --------------------------------------------------
    if (commodity in SVR_RESULTS and dist in SVR_RESULTS[commodity] and
            commodity in ml_datasets and dist in ml_datasets[commodity]):
        data   = ml_datasets[commodity][dist]
        X, y_t = data['X'], data['y']
        sc     = SVR_RESULTS[commodity][dist]['scaler']
        svr_m  = SVR_RESULTS[commodity][dist]['model']
        fit    = svr_m.predict(sc.transform(X))
        n_fit  = len(fit)
        fd     = dates_act[-n_fit:] if n_fit <= len(dates_act) else dates_act
        ax_l.plot(fd, fit[:len(fd)], color='#E91E8C',
                  linestyle='--', linewidth=1.5, label='SVR', alpha=0.85)
        min_len = min(len(y_t), len(fit))
        residuals['SVR'] = (y_t[:min_len] - fit[:min_len],
                            '#E91E8C')

    # --- RF in-sample fit ---------------------------------------------------
    if (commodity in RF_RESULTS and dist in RF_RESULTS[commodity] and
            commodity in ml_datasets and dist in ml_datasets[commodity]):
        data   = ml_datasets[commodity][dist]
        X, y_t = data['X'], data['y']
        rf_m   = RF_RESULTS[commodity][dist]['model']
        fit    = rf_m.predict(X)
        n_fit  = len(fit)
        fd     = dates_act[-n_fit:] if n_fit <= len(dates_act) else dates_act
        ax_l.plot(fd, fit[:len(fd)], color='#4CAF50',
                  linestyle='--', linewidth=1.5, label='Random Forest', alpha=0.85)
        min_len = min(len(y_t), len(fit))
        residuals['RF'] = (y_t[:min_len] - fit[:min_len], '#4CAF50')

    # --- LSTM in-sample fit -------------------------------------------------
    if commodity in LSTM_RESULTS and dist in LSTM_RESULTS[commodity]:
        lr      = LSTM_RESULTS[commodity][dist]
        y_full  = lr['y_full']
        y_range = lr['y_max'] - lr['y_min'] + 1e-8
        y_sc    = (y_full - lr['y_min']) / y_range
        X_seq, _ = make_lstm_sequences(y_sc, WINDOW)
        if len(X_seq) > 0:
            pred_sc = lr['model'].predict(X_seq, verbose=0).flatten()
            fit     = pred_sc * y_range + lr['y_min']
            n_fit   = len(fit)
            fd      = dates_act[-n_fit:] if n_fit <= len(dates_act) else dates_act
            ax_l.plot(fd, fit[:len(fd)], color='#00BCD4',
                      linestyle='-.', linewidth=1.7, label='LSTM', alpha=0.85)
            y_lstm_act = y_full[WINDOW:WINDOW+n_fit]
            min_len = min(len(y_lstm_act), len(fit))
            residuals['LSTM'] = (y_lstm_act[:min_len] - fit[:min_len],
                                 '#00BCD4')

    ax_l.set_title(f'{commodity}  /  {dist}  —  In-Sample Fit',
                   color=col, fontsize=9.5, fontweight='bold')
    ax_l.set_ylabel('UGX / kg', fontsize=8)
    ax_l.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax_l.legend(fontsize=7.5, loc='upper left', framealpha=0.35, ncol=2)
    ax_l.grid(True, alpha=0.18)

    # --- Right: residual bars -----------------------------------------------
    if residuals:
        offsets  = {'SVR': -0.25, 'RF': 0.0, 'LSTM': 0.25}
        width    = 0.22
        rmse_parts = []
        for mname, (resid, mc) in residuals.items():
            x_pos = np.arange(len(resid)) + offsets.get(mname, 0)
            ax_r.bar(x_pos, resid, width=width, color=mc,
                     alpha=0.72, label=mname, edgecolor='none')
            rmse_parts.append(f"{mname}={np.sqrt(np.mean(resid**2)):,.0f}")
        ax_r.axhline(0, color='white', linewidth=0.8, alpha=0.5)
        ax_r.set_title(f'{commodity}  /  {dist}  —  Residuals  (Actual − Predicted)',
                       color=col, fontsize=9.5, fontweight='bold')
        ax_r.set_ylabel('Residual UGX/kg', fontsize=8)
        ax_r.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:+,.0f}'))
        ax_r.legend(fontsize=7.5, loc='upper right', framealpha=0.35)
        ax_r.grid(True, alpha=0.18, axis='y')
        ax_r.text(0.01, 0.97, 'RMSE  —  ' + '   '.join(rmse_parts),
                  transform=ax_r.transAxes, fontsize=6.5,
                  color='#C8D8E8', va='top')
    else:
        ax_r.text(0.5, 0.5, 'ML models not fitted\\nfor this district',
                  ha='center', va='center', transform=ax_r.transAxes,
                  color='#8A9BAB', fontsize=9)
        ax_r.axis('off')

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/fig15_ml_insample_fit.png', dpi=130,
            bbox_inches='tight', facecolor='#0A1520')
plt.show()
plt.close()
print('Figure 15 saved: outputs/fig15_ml_insample_fit.png')
"""

# ===========================================================================
# FIX 4 — Cell 44: Figure 16 — Seasonal pattern comparison (robust)
# ===========================================================================
FIG16 = """\
# ============================================================================
# FIGURE 16: SEASONAL PATTERN COMPARISON — SARIMA vs SVR vs RF vs LSTM
# ============================================================================
# Each model encodes seasonality differently:
#   SARIMA  — explicit seasonal differencing (D=1, s=12) forces the model
#             to learn ONE fixed annual harvest cycle from the data.
#   SVR     — sin/cos month features + lag12 let the RBF kernel interpolate
#             between similar seasonal states smoothly.
#   RF      — tree splits partition the month-cycle non-linearly; can capture
#             asymmetric lean vs. post-harvest patterns.
#   LSTM    — no explicit seasonal features; the recurrent gates must DISCOVER
#             the 12-month pattern from raw price sequences alone.
#
# SEASONAL PRICE INDEX (SPI):
#   SPI(month m) = forecast(m) / annual_average * 100
#   100 = neutral   >110 = lean season peak   <95 = post-harvest surplus
#
# LEFT  — Line chart comparing SPI curves across all 4 models per commodity.
# RIGHT — SPI heatmap (model rows x month cols) — colour: red=high, green=low.
# ----------------------------------------------------------------------------

month_lbls = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
SPI_COLORS = {'SARIMA':'#FFD700','SVR':'#E91E8C',
              'RF':'#4CAF50','LSTM':'#00BCD4'}

n_comm = len(COMMODITIES)
fig, axes = plt.subplots(n_comm, 2, figsize=(26, n_comm * 4.0))
fig.patch.set_facecolor('#0A1520')
fig.suptitle('Seasonal Pattern Analysis — How Each Model Reads the Harvest Cycle  (2026)',
             fontsize=14, fontweight='bold', color='white', y=1.002)

def nat_fc(source_dict, commodity, key=None):
    # Average forecast array across all available districts for a model.
    districts = list(source_dict.get(commodity, {}).keys())
    vals = []
    for d in districts:
        entry = source_dict[commodity][d]
        fc = entry[key] if key else entry.get('forecast')
        if fc is not None and len(fc) == 12:
            vals.append(fc)
    return np.mean(vals, axis=0) if vals else None

def to_spi(fc_arr):
    ann = fc_arr.mean()
    return fc_arr / ann * 100 if ann > 0 else fc_arr

for ci, commodity in enumerate(COMMODITIES):
    col    = PALETTE.get(commodity, '#888888')
    ax_l   = axes[ci, 0]
    ax_r   = axes[ci, 1]
    ax_l.set_facecolor('#0F1923')
    ax_r.set_facecolor('#0F1923')

    spi_models = {}

    # SARIMA blended
    fc = nat_fc(blended_forecasts, commodity, 'forecast')
    if fc is not None:
        spi_models['SARIMA'] = to_spi(fc)

    # SVR
    svr_vals = [ml_forecasts[commodity][d]['SVR']
                for d in ml_forecasts.get(commodity, {})
                if ml_forecasts[commodity][d].get('SVR') is not None
                and len(ml_forecasts[commodity][d]['SVR']) == 12]
    if svr_vals:
        spi_models['SVR'] = to_spi(np.mean(svr_vals, axis=0))

    # RF
    rf_vals = [ml_forecasts[commodity][d]['RF']
               for d in ml_forecasts.get(commodity, {})
               if ml_forecasts[commodity][d].get('RF') is not None
               and len(ml_forecasts[commodity][d]['RF']) == 12]
    if rf_vals:
        spi_models['RF'] = to_spi(np.mean(rf_vals, axis=0))

    # LSTM (key districts only)
    lstm_vals = [ml_forecasts[commodity][d]['LSTM']
                 for d in ml_forecasts.get(commodity, {})
                 if ml_forecasts[commodity][d].get('LSTM') is not None
                 and len(ml_forecasts[commodity][d]['LSTM']) == 12]
    if lstm_vals:
        spi_models['LSTM'] = to_spi(np.mean(lstm_vals, axis=0))

    x = np.arange(12)

    # --- Left: SPI line chart -----------------------------------------------
    for mname, spi in spi_models.items():
        ls = '-' if mname == 'SARIMA' else '--'
        lw = 2.4 if mname == 'SARIMA' else 1.6
        ax_l.plot(x, spi, color=SPI_COLORS[mname], linestyle=ls,
                  linewidth=lw, marker='o', markersize=3.5,
                  label=mname, alpha=0.9)

    # Reference bands
    ax_l.axhspan(110, 130, color='#FF4444', alpha=0.06, label='Lean >110')
    ax_l.axhspan(70,   95, color='#4CAF50', alpha=0.06, label='Surplus <95')
    ax_l.axhline(100, color='white', linewidth=0.7, alpha=0.35, linestyle=':')

    ax_l.set_xticks(x)
    ax_l.set_xticklabels(month_lbls, fontsize=8)
    ax_l.set_ylim(80, 125)
    ax_l.set_title(f'{commodity} — Seasonal Price Index by Model  (2026 forecast)',
                   color=col, fontsize=9.5, fontweight='bold')
    ax_l.set_ylabel('SPI  (100 = annual average)', fontsize=8)
    ax_l.legend(fontsize=7.5, loc='upper right', ncol=2, framealpha=0.35)
    ax_l.grid(True, alpha=0.18)

    # --- Right: SPI heatmap -------------------------------------------------
    if spi_models:
        model_names = list(spi_models.keys())
        matrix = np.array([spi_models[m] for m in model_names])
        im = ax_r.imshow(matrix, cmap='RdYlGn_r', vmin=88, vmax=116, aspect='auto')
        ax_r.set_xticks(range(12))
        ax_r.set_xticklabels(month_lbls, fontsize=7.5, rotation=40, ha='right')
        ax_r.set_yticks(range(len(model_names)))
        ax_r.set_yticklabels(model_names, fontsize=9)
        for ri in range(len(model_names)):
            for ci2 in range(12):
                v = matrix[ri, ci2]
                tc = 'white' if (v > 110 or v < 93) else '#111111'
                ax_r.text(ci2, ri, f'{v:.1f}', ha='center', va='center',
                          fontsize=7, fontweight='bold', color=tc)
        plt.colorbar(im, ax=ax_r, shrink=0.82, label='SPI')
        ax_r.set_title(f'{commodity} — SPI Heatmap: Model vs Month',
                       color=col, fontsize=9.5, fontweight='bold')
    else:
        ax_r.text(0.5, 0.5, 'No ML forecasts available',
                  ha='center', va='center', transform=ax_r.transAxes,
                  color='#8A9BAB', fontsize=9)
        ax_r.axis('off')

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/fig16_seasonal_model_comparison.png', dpi=130,
            bbox_inches='tight', facecolor='#0A1520')
plt.show()
plt.close()
print('Figure 16 saved: outputs/fig16_seasonal_model_comparison.png')
"""

# ===========================================================================
# FIX 5 — Cell 45: Figure 17 — District price spread (robust)
# ===========================================================================
FIG17 = """\
# ============================================================================
# FIGURE 17: DISTRICT PRICE SPREAD — Model Forecasts Across All Markets  (2026)
# ============================================================================
# Each model produces a different district-level price spread for 2026:
#
#   SARIMA   — uses district-specific calibrated price multipliers.
#              The spread reflects real historical market price gaps.
#   SVR/RF   — uses per-district lag and seasonal feature vectors.
#              Spread depends on how different each district's price history is.
#   LSTM     — trained on key districts only; spread is among those 8-11 markets.
#   Ensemble — inverse-MAPE weighted average; shrinks toward better-fitting models.
#
# LEFT  — Box plot: distribution of annual avg 2026 forecasts across districts.
#          Box = IQR (25th–75th pct)  |  Whiskers = min/max  |  Line = median
#          Wider box → model sees bigger market price gaps.
#
# RIGHT — Line chart: annual avg forecast per district, coloured by model.
#          Lets you see exactly which markets each model rates as expensive
#          (Kampala premium) vs cheap (rural surplus markets).
# ----------------------------------------------------------------------------

MODEL_PALETTE = {
    'SARIMA':  ('#FFD700', '-',  2.2),
    'SVR':     ('#E91E8C', '--', 1.5),
    'RF':      ('#4CAF50', '--', 1.5),
    'LSTM':    ('#00BCD4', '-.', 1.7),
    'Ensemble':('#FFFFFF', '-',  2.0),
}

n_comm = len(COMMODITIES)
fig, axes = plt.subplots(n_comm, 2, figsize=(28, n_comm * 4.2))
fig.patch.set_facecolor('#0A1520')
fig.suptitle('District Price Spread per Model  —  2026 Annual Average Forecast',
             fontsize=14, fontweight='bold', color='white', y=1.002)

for ci, commodity in enumerate(COMMODITIES):
    col       = PALETTE.get(commodity, '#888888')
    districts = COMMODITY_DISTRICTS[commodity]
    ax_box    = axes[ci, 0]
    ax_line   = axes[ci, 1]
    ax_box.set_facecolor('#0F1923')
    ax_line.set_facecolor('#0F1923')

    model_avgs  = {}   # model_name -> list of annual avg per district

    for d in districts:
        sarima_fc = blended_forecasts.get(commodity, {}).get(d, {}).get('forecast')
        if sarima_fc is not None:
            model_avgs.setdefault('SARIMA', []).append(float(sarima_fc.mean()))

        ml_entry = ml_forecasts.get(commodity, {}).get(d, {})
        for key in ['SVR', 'RF', 'LSTM', 'Ensemble']:
            fc = ml_entry.get(key)
            if fc is not None and len(fc) == 12:
                model_avgs.setdefault(key, []).append(float(fc.mean()))

    if not model_avgs:
        for ax in [ax_box, ax_line]:
            ax.text(0.5, 0.5, 'No forecasts available',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#8A9BAB', fontsize=9)
            ax.axis('off')
        continue

    # --- Left: box plot -----------------------------------------------------
    ordered_models = [m for m in MODEL_PALETTE if m in model_avgs]
    box_data  = [model_avgs[m] for m in ordered_models]
    box_cols  = [MODEL_PALETTE[m][0] for m in ordered_models]

    bp = ax_box.boxplot(
        box_data, patch_artist=True, widths=0.55,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color='#8A9BAB', linewidth=1),
        capprops=dict(color='#8A9BAB', linewidth=1),
        flierprops=dict(marker='o', markersize=4,
                        markerfacecolor='white', alpha=0.5, linestyle='none')
    )
    for patch, bc in zip(bp['boxes'], box_cols):
        patch.set_facecolor(bc)
        patch.set_alpha(0.65)

    ax_box.set_xticks(range(1, len(ordered_models)+1))
    ax_box.set_xticklabels(ordered_models, fontsize=8)
    ax_box.set_title(f'{commodity} — District Annual Avg Distribution per Model',
                     color=col, fontsize=9.5, fontweight='bold')
    ax_box.set_ylabel('Annual Avg Forecast  UGX/kg', fontsize=8)
    ax_box.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax_box.grid(True, alpha=0.18, axis='y')

    # Annotate IQR
    for bi, (mname, avgs) in enumerate([(m, model_avgs[m]) for m in ordered_models], 1):
        arr = np.array(avgs)
        if len(arr) >= 4:
            iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
            ax_box.text(bi, np.percentile(arr, 75) + arr.std()*0.05,
                        f'{iqr:,.0f}', ha='center', va='bottom',
                        fontsize=6.5, color='white', alpha=0.8)

    # --- Right: per-district annual avg line --------------------------------
    x = np.arange(len(districts))
    has_line = False
    for mname, (mc, ls, lw) in MODEL_PALETTE.items():
        vals = []
        for d in districts:
            if mname == 'SARIMA':
                fc = blended_forecasts.get(commodity, {}).get(d, {}).get('forecast')
            else:
                fc = ml_forecasts.get(commodity, {}).get(d, {}).get(mname)
            vals.append(float(fc.mean()) if (fc is not None and len(fc)==12) else np.nan)

        if not all(np.isnan(vals)):
            ax_line.plot(x, vals, color=mc, linestyle=ls, linewidth=lw,
                         marker='o', markersize=3.5, label=mname, alpha=0.88)
            has_line = True

    if has_line:
        ax_line.set_xticks(x)
        ax_line.set_xticklabels(districts, fontsize=6, rotation=55, ha='right')
        ax_line.set_title(f'{commodity} — 2026 Annual Avg by District & Model',
                          color=col, fontsize=9.5, fontweight='bold')
        ax_line.set_ylabel('Annual Avg UGX/kg', fontsize=8)
        ax_line.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax_line.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.35)
        ax_line.grid(True, alpha=0.18)

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/fig17_district_spread.png', dpi=130,
            bbox_inches='tight', facecolor='#0A1520')
plt.show()
plt.close()
print('Figure 17 saved: outputs/fig17_district_spread.png')
"""

# Apply cell rewrites
cells[43]['source'] = [FIG15]
cells[44]['source'] = [FIG16]
cells[45]['source'] = [FIG17]
print('Cells 43/44/45 rewritten (Fig 15/16/17)')

# ===========================================================================
# VERIFY — check no remaining known bad patterns
# ===========================================================================
problems = []
for i, c in enumerate(cells):
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source'])
    if 'subplots(3, 8,' in src and 'enumerate(COMMODITIES)' in src:
        problems.append(f'Cell {i}: subplots(3,8) with COMMODITIES loop')
    if 'backtest_metrics' in src:
        problems.append(f'Cell {i}: stale backtest_metrics')

if problems:
    for p in problems: print('WARNING:', p)
else:
    print('Verification: no remaining known errors')

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nNotebook saved. Total cells: {len(cells)}')
print('Summary of all fixes applied:')
print('  [1] Cell 22: subplots(3,8) -> subplots(n_comm,8)  [7 commodities]')
print('  [2] Cell 23: same fix')
print('  [3] Cell 27: fc array guard verified')
print('  [4] Cell 43: Figure 15 fully rewritten — robust in-sample fit chart')
print('  [5] Cell 44: Figure 16 fully rewritten — robust seasonal SPI comparison')
print('  [6] Cell 45: Figure 17 fully rewritten — robust district spread chart')
