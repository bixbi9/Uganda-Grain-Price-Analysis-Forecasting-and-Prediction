"""
patch_ml_charts.py
Inserts three new analytical chart cells into Section 11 (before Appendix):

  Figure 15 — ML In-Sample Fit:  actual vs SVR / RF / LSTM fitted values
               per commodity, showing which model best tracks historical prices.

  Figure 16 — Seasonal Pattern Comparison:  12-month seasonal index
               for each model (SARIMA / SVR / RF / LSTM) per commodity,
               revealing how each technique interprets the harvest cycle.

  Figure 17 — Price Spread & Volatility:  district-level price spread in
               2026 forecasts across models (box chart per commodity),
               showing which model produces tighter / wider uncertainty.
"""
import json, sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

NB = 'Uganda_Grain_SARIMA_Analysis.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def code_cell(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[src]}

# =============================================================================
# FIGURE 15 — In-sample fit: actual vs ML models
# =============================================================================
FIG15 = """\
# ============================================================================
# FIGURE 15: ML MODEL IN-SAMPLE FIT — Actual vs SVR / RF / LSTM
# ============================================================================
#
# Shows how well each ML model tracks the historical 2024-2025 monthly prices
# for a representative district (Kampala for main commodities, Kabale for Barley).
#
# WHY THIS MATTERS:
#   A model that fits historical data well (low residuals) is not automatically
#   the best forecaster — but extreme misfits reveal poor feature calibration.
#   Comparing fitted curves tells us:
#     * SVR — does the kernel margin capture the price level correctly?
#     * RF  — do the tree splits isolate seasonal jumps cleanly?
#     * LSTM — does the recurrent memory track multi-month price trajectories?
#
# Layout: 7 rows (one per commodity) x 2 panels
#   Left  — fitted price lines overlaid on actual prices
#   Right — residual errors (actual - predicted) per model
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

MODEL_LINE = {
    'Actual': ('#FFFFFF', '-',  2.4),
    'SVR':    ('#E91E8C', '--', 1.5),
    'RF':     ('#4CAF50', '--', 1.5),
    'LSTM':   ('#00BCD4', '-.',  1.7),
}

n_comm = len(COMMODITIES)
fig, axes = plt.subplots(n_comm, 2, figsize=(26, n_comm * 4.2))
fig.patch.set_facecolor('#0A1520')
fig.suptitle('ML Model In-Sample Fit — Actual Prices vs SVR / Random Forest / LSTM (2024-2025)',
             fontsize=14, fontweight='bold', color='white', y=1.002)

for ci, commodity in enumerate(COMMODITIES):
    col  = PALETTE.get(commodity, '#888888')
    dist = REPR_DISTRICT.get(commodity, 'Kampala')

    ax_fit = axes[ci, 0]
    ax_res = axes[ci, 1]
    ax_fit.set_facecolor('#0F1923')
    ax_res.set_facecolor('#0F1923')

    # ---- Get actual monthly prices for representative district --------------
    mask = ((df_monthly['Commodity'] == commodity) &
            (df_monthly['District']  == dist))
    sub  = df_monthly[mask].sort_values('Date')
    price_col = ('Avg_Monthly_Price' if 'Avg_Monthly_Price' in sub.columns
                 else 'Price_UGX')

    if len(sub) < 6:
        ax_fit.set_title(f'{commodity} / {dist} — insufficient data',
                         color=col, fontsize=9)
        ax_res.axis('off')
        continue

    dates_actual = sub['Date'].values
    y_actual     = sub[price_col].values.astype(float)

    # Plot actual
    ax_fit.plot(dates_actual, y_actual, **dict(zip(
        ['color','linestyle','linewidth'], MODEL_LINE['Actual'])),
        label='Actual', zorder=6)

    residuals_dict = {}

    # ---- SVR in-sample prediction ------------------------------------------
    if (commodity in SVR_RESULTS and dist in SVR_RESULTS[commodity] and
            commodity in ml_datasets and dist in ml_datasets[commodity]):
        data    = ml_datasets[commodity][dist]
        X, y_t  = data['X'], data['y']
        scaler  = SVR_RESULTS[commodity][dist]['scaler']
        svr_m   = SVR_RESULTS[commodity][dist]['model']
        X_sc    = scaler.transform(X)
        svr_fit = svr_m.predict(X_sc)
        # align with dates (features start from index 12)
        fit_dates = dates_actual[12:]
        ax_fit.plot(fit_dates, svr_fit, **dict(zip(
            ['color','linestyle','linewidth'], MODEL_LINE['SVR'])),
            label='SVR', alpha=0.85)
        residuals_dict['SVR'] = y_t - svr_fit

    # ---- RF in-sample prediction -------------------------------------------
    if (commodity in RF_RESULTS and dist in RF_RESULTS[commodity] and
            commodity in ml_datasets and dist in ml_datasets[commodity]):
        data   = ml_datasets[commodity][dist]
        X, y_t = data['X'], data['y']
        rf_m   = RF_RESULTS[commodity][dist]['model']
        rf_fit = rf_m.predict(X)
        fit_dates = dates_actual[12:]
        ax_fit.plot(fit_dates, rf_fit, **dict(zip(
            ['color','linestyle','linewidth'], MODEL_LINE['RF'])),
            label='Random Forest', alpha=0.85)
        residuals_dict['RF'] = y_t - rf_fit

    # ---- LSTM in-sample prediction -----------------------------------------
    if (commodity in LSTM_RESULTS and dist in LSTM_RESULTS[commodity]):
        lstm_res = LSTM_RESULTS[commodity][dist]
        y_full   = lstm_res['y_full']
        y_min_v  = lstm_res['y_min']
        y_max_v  = lstm_res['y_max']
        y_range  = y_max_v - y_min_v + 1e-8
        y_sc_full = (y_full - y_min_v) / y_range

        X_lstm, y_lstm_t = make_lstm_sequences(y_sc_full, WINDOW)
        if len(X_lstm) > 0:
            pred_sc   = lstm_res['model'].predict(X_lstm, verbose=0).flatten()
            lstm_fit  = pred_sc * y_range + y_min_v
            fit_dates = dates_actual[WINDOW:]
            ax_fit.plot(fit_dates, lstm_fit, **dict(zip(
                ['color','linestyle','linewidth'], MODEL_LINE['LSTM'])),
                label='LSTM', alpha=0.85)
            # align residuals to same window
            y_lstm_actual = y_full[WINDOW:]
            if len(y_lstm_actual) == len(lstm_fit):
                residuals_dict['LSTM'] = y_lstm_actual - lstm_fit

    ax_fit.set_title(f'{commodity} / {dist} — In-Sample Fit',
                     color=col, fontsize=9.5, fontweight='bold')
    ax_fit.set_ylabel('UGX / kg', fontsize=8)
    ax_fit.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax_fit.legend(fontsize=7, loc='upper left', framealpha=0.35)
    ax_fit.grid(True, alpha=0.18)

    # ---- Right panel: residual bars ----------------------------------------
    res_colors = {'SVR': '#E91E8C', 'RF': '#4CAF50', 'LSTM': '#00BCD4'}
    has_res = False
    for mi, (mname, resid) in enumerate(residuals_dict.items()):
        if resid is None or len(resid) == 0:
            continue
        x_pos = np.arange(len(resid)) + mi * 0.25
        ax_res.bar(x_pos, resid, width=0.22,
                   color=res_colors.get(mname, '#888888'),
                   alpha=0.70, label=mname, edgecolor='none')
        has_res = True

    if has_res:
        ax_res.axhline(0, color='white', linewidth=0.8, alpha=0.5)
        ax_res.set_title(f'{commodity} / {dist} — Residuals (Actual - Predicted)',
                         color=col, fontsize=9.5, fontweight='bold')
        ax_res.set_ylabel('Residual (UGX/kg)', fontsize=8)
        ax_res.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:+,.0f}'))
        ax_res.legend(fontsize=7, loc='upper right', framealpha=0.35)
        ax_res.grid(True, alpha=0.18, axis='y')

        # RMSE annotation per model
        rmse_str = '  '.join(
            f"{mn}: RMSE={np.sqrt(np.mean(r**2)):,.0f}"
            for mn, r in residuals_dict.items() if r is not None and len(r) > 0
        )
        ax_res.text(0.01, 0.97, rmse_str,
                    transform=ax_res.transAxes, fontsize=6.5,
                    color='#C8D8E8', va='top')
    else:
        ax_res.axis('off')

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/fig15_ml_insampled_fit.png', dpi=130,
            bbox_inches='tight', facecolor='#0A1520')
plt.show()
plt.close()
print('Figure 15 saved: outputs/fig15_ml_insampled_fit.png')
"""

# =============================================================================
# FIGURE 16 — Seasonal pattern comparison across models
# =============================================================================
FIG16 = """\
# ============================================================================
# FIGURE 16: SEASONAL PATTERN COMPARISON — SARIMA vs SVR vs RF vs LSTM
# ============================================================================
#
# Each model has a different way of encoding seasonality:
#   SARIMA  — explicit seasonal differencing (D=1, s=12) + seasonal AR/MA terms.
#             Captures ONE fixed annual harvest cycle learned from the data.
#   SVR     — seasonal signal enters via sin/cos month features and lag12.
#             The RBF kernel smooths across similar seasonal states.
#   RF      — seasonal signal enters via the same 9 features.
#             Decision tree splits partition the month-space non-linearly,
#             potentially capturing asymmetric lean vs. post-harvest patterns.
#   LSTM    — no explicit seasonal features; the recurrent memory must LEARN
#             the 12-month cycle from the raw price sequence alone.
#             With only 24 months of training, the LSTM seasonal signal
#             may be underfit compared to the feature-based models.
#
# Chart: For each commodity, compute a Seasonal Price Index (SPI) from the
# 2026 forecast of each model:
#   SPI(month m) = forecast(m) / annual_average * 100
#   100 = neutral  |  >110 = lean season  |  <95 = post-harvest surplus
#
# This allows a direct comparison of HOW EACH MODEL READS THE SEASONS,
# independent of absolute price level differences.
# ----------------------------------------------------------------------------

month_lbls = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']

SPI_MODEL_COLORS = {
    'SARIMA': '#FFD700',
    'SVR':    '#E91E8C',
    'RF':     '#4CAF50',
    'LSTM':   '#00BCD4',
}

n_comm = len(COMMODITIES)
fig, axes = plt.subplots(n_comm, 2, figsize=(26, n_comm * 4.0))
fig.patch.set_facecolor('#0A1520')
fig.suptitle('Seasonal Pattern Analysis — How Each Model Reads the Harvest Cycle',
             fontsize=14, fontweight='bold', color='white', y=1.002)

for ci, commodity in enumerate(COMMODITIES):
    col       = PALETTE.get(commodity, '#888888')
    districts = list(ml_forecasts.get(commodity, {}).keys())
    ax_spi    = axes[ci, 0]
    ax_heat   = axes[ci, 1]
    ax_spi.set_facecolor('#0F1923')
    ax_heat.set_facecolor('#0F1923')

    spi_by_model = {}

    # ---- SARIMA SPI ---------------------------------------------------------
    if blended_forecasts.get(commodity):
        fc_vals = [blended_forecasts[commodity][d]['forecast']
                   for d in districts if d in blended_forecasts[commodity]]
        if fc_vals:
            fc_nat = np.mean(fc_vals, axis=0)
            ann    = fc_nat.mean()
            spi_by_model['SARIMA'] = fc_nat / ann * 100

    # ---- SVR SPI ------------------------------------------------------------
    svr_fcs = [ml_forecasts[commodity][d]['SVR']
               for d in districts
               if d in ml_forecasts.get(commodity, {})
               and ml_forecasts[commodity][d].get('SVR') is not None]
    if svr_fcs:
        fc_nat = np.mean(svr_fcs, axis=0)
        ann    = fc_nat.mean()
        spi_by_model['SVR'] = fc_nat / ann * 100

    # ---- RF SPI -------------------------------------------------------------
    rf_fcs = [ml_forecasts[commodity][d]['RF']
              for d in districts
              if d in ml_forecasts.get(commodity, {})
              and ml_forecasts[commodity][d].get('RF') is not None]
    if rf_fcs:
        fc_nat = np.mean(rf_fcs, axis=0)
        ann    = fc_nat.mean()
        spi_by_model['RF'] = fc_nat / ann * 100

    # ---- LSTM SPI -----------------------------------------------------------
    lstm_fcs = [ml_forecasts[commodity][d]['LSTM']
                for d in districts
                if d in ml_forecasts.get(commodity, {})
                and ml_forecasts[commodity][d].get('LSTM') is not None]
    if lstm_fcs:
        fc_nat = np.mean(lstm_fcs, axis=0)
        ann    = fc_nat.mean()
        spi_by_model['LSTM'] = fc_nat / ann * 100

    # ---- Left: SPI line chart -----------------------------------------------
    x = np.arange(12)
    for mname, spi in spi_by_model.items():
        mc, ls, lw = SPI_MODEL_COLORS[mname], '--', 1.6
        if mname == 'SARIMA':
            ls, lw = '-', 2.4
        ax_spi.plot(x, spi, color=mc, linestyle=ls, linewidth=lw,
                    label=mname, marker='o', markersize=3.5, alpha=0.9)

    # Reference lines
    ax_spi.axhline(110, color='#FF4444', linestyle=':', linewidth=0.9,
                   alpha=0.55, label='Lean (>110)')
    ax_spi.axhline(100, color='white',   linestyle=':', linewidth=0.7, alpha=0.4)
    ax_spi.axhline(95,  color='#4CAF50', linestyle=':', linewidth=0.9,
                   alpha=0.55, label='Surplus (<95)')

    # Shade lean months
    lean_flag = False
    for m_idx, m_lbl in enumerate(month_lbls):
        if any(spi[m_idx] > 108 for spi in spi_by_model.values() if spi is not None):
            ax_spi.axvspan(m_idx-0.4, m_idx+0.4, color='#FF4444', alpha=0.06)
            lean_flag = True

    ax_spi.set_xticks(x)
    ax_spi.set_xticklabels(month_lbls, fontsize=8)
    ax_spi.set_title(f'{commodity} — Seasonal Price Index by Model (2026)',
                     color=col, fontsize=9.5, fontweight='bold')
    ax_spi.set_ylabel('SPI (100 = annual avg)', fontsize=8)
    ax_spi.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.35)
    ax_spi.grid(True, alpha=0.18)
    if lean_flag:
        ax_spi.text(0.01, 0.98, 'Light red shading = lean season months',
                    transform=ax_spi.transAxes, fontsize=6.5,
                    color='#FF6B6B', va='top')

    # ---- Right: SPI heatmap (models x months) --------------------------------
    if spi_by_model:
        model_names_heat = list(spi_by_model.keys())
        matrix = np.array([spi_by_model[m] for m in model_names_heat])

        im = ax_heat.imshow(matrix, cmap='RdYlGn_r',
                            vmin=88, vmax=115, aspect='auto')
        ax_heat.set_xticks(range(12))
        ax_heat.set_xticklabels(month_lbls, fontsize=7.5, rotation=40, ha='right')
        ax_heat.set_yticks(range(len(model_names_heat)))
        ax_heat.set_yticklabels(model_names_heat, fontsize=9)

        for ri in range(len(model_names_heat)):
            for ci2 in range(12):
                val = matrix[ri, ci2]
                txt_col = 'white' if (val > 110 or val < 93) else '#1A2A3A'
                ax_heat.text(ci2, ri, f'{val:.1f}',
                             ha='center', va='center',
                             fontsize=7, fontweight='bold', color=txt_col)

        plt.colorbar(im, ax=ax_heat, shrink=0.85,
                     label='SPI (100 = annual avg)')
        ax_heat.set_title(f'{commodity} — SPI Heatmap: Model vs Month',
                          color=col, fontsize=9.5, fontweight='bold')
    else:
        ax_heat.axis('off')

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/fig16_seasonal_model_comparison.png', dpi=130,
            bbox_inches='tight', facecolor='#0A1520')
plt.show()
plt.close()
print('Figure 16 saved: outputs/fig16_seasonal_model_comparison.png')
"""

# =============================================================================
# FIGURE 17 — Price spread & district volatility across models
# =============================================================================
FIG17 = """\
# ============================================================================
# FIGURE 17: DISTRICT PRICE SPREAD — How Each Model Forecasts Market Variation
# ============================================================================
#
# Different models produce different levels of district-level price spread:
#   * SARIMA uses district-specific calibrated multipliers — the spread
#     reflects real historical price gaps between markets.
#   * SVR / RF / GB use lag and seasonal features per district — the spread
#     depends on how different each district's price history is.
#   * LSTM uses per-district scaled sequences — the spread may compress
#     because LSTM normalises prices to [0,1] before learning.
#
# Chart layout: 7 rows (one per commodity) x 2 panels
#   Left  — Box plot of the 12-month average forecast per district,
#            grouped by model. Wider box = higher model uncertainty.
#   Right — Line chart showing annual average 2026 forecast per district
#            for each model, with Kampala/Gulu as reference anchors.
#
# READING THE CHART:
#   * Wide box (large IQR) = the model forecasts big district price gaps
#     → useful for targeting high-price markets for sales
#   * Narrow box          = the model is uncertain about district variation
#   * Outlier dots        = extreme market (typically Kampala premium or
#     rural discount)
# ----------------------------------------------------------------------------

MODEL_BOX_COLORS = {
    'SARIMA':  '#FFD700',
    'SVR':     '#E91E8C',
    'RF':      '#4CAF50',
    'LSTM':    '#00BCD4',
    'Ensemble':'#FFFFFF',
}

n_comm = len(COMMODITIES)
fig, axes = plt.subplots(n_comm, 2, figsize=(26, n_comm * 4.2))
fig.patch.set_facecolor('#0A1520')
fig.suptitle(
    'District Price Spread per Model — How Models Forecast Market Variation (2026)',
    fontsize=14, fontweight='bold', color='white', y=1.002)

for ci, commodity in enumerate(COMMODITIES):
    col       = PALETTE.get(commodity, '#888888')
    districts = list(ml_forecasts.get(commodity, {}).keys())
    ax_box    = axes[ci, 0]
    ax_line   = axes[ci, 1]
    ax_box.set_facecolor('#0F1923')
    ax_line.set_facecolor('#0F1923')

    # Collect annual average forecast per district for each model
    model_district_avgs = {}

    # SARIMA
    sarima_avgs = []
    for d in districts:
        fc = blended_forecasts.get(commodity, {}).get(d, {}).get('forecast')
        if fc is not None:
            sarima_avgs.append(fc.mean())
    if sarima_avgs:
        model_district_avgs['SARIMA'] = sarima_avgs

    # SVR / RF / LSTM / Ensemble
    for key in ['SVR', 'RF', 'LSTM', 'Ensemble']:
        vals = []
        for d in districts:
            fc = ml_forecasts.get(commodity, {}).get(d, {}).get(key)
            if fc is not None:
                vals.append(fc.mean())
        if vals:
            model_district_avgs[key] = vals

    if not model_district_avgs:
        ax_box.axis('off')
        ax_line.axis('off')
        continue

    # ---- Left: box plot per model -------------------------------------------
    box_data   = []
    box_labels = []
    box_cols   = []
    for mname, avgs in model_district_avgs.items():
        box_data.append(avgs)
        box_labels.append(mname)
        box_cols.append(MODEL_BOX_COLORS.get(mname, '#888888'))

    bp = ax_box.boxplot(box_data, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color='#8A9BAB'),
                        capprops=dict(color='#8A9BAB'),
                        flierprops=dict(marker='o', markersize=4,
                                        markerfacecolor='white', alpha=0.5))
    for patch, bc in zip(bp['boxes'], box_cols):
        patch.set_facecolor(bc)
        patch.set_alpha(0.65)

    ax_box.set_xticklabels(box_labels, fontsize=8)
    ax_box.set_title(f'{commodity} — District Avg Price Distribution per Model',
                     color=col, fontsize=9.5, fontweight='bold')
    ax_box.set_ylabel('Annual Avg Forecast UGX/kg', fontsize=8)
    ax_box.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax_box.grid(True, alpha=0.18, axis='y')

    # Annotate IQR width
    for bi, (mname, avgs) in enumerate(model_district_avgs.items(), 1):
        arr = np.array(avgs)
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        ax_box.text(bi, np.percentile(arr, 75),
                    f'IQR\n{iqr:,.0f}', ha='center', va='bottom',
                    fontsize=6, color='white', alpha=0.75)

    # ---- Right: per-district annual avg line chart -------------------------
    x = np.arange(len(districts))
    for mname, mc in MODEL_BOX_COLORS.items():
        vals = []
        for d in districts:
            if mname == 'SARIMA':
                fc = blended_forecasts.get(commodity, {}).get(d, {}).get('forecast')
            else:
                fc = ml_forecasts.get(commodity, {}).get(d, {}).get(mname)
            vals.append(fc.mean() if fc is not None else np.nan)

        if not all(np.isnan(vals)):
            ls = '-' if mname in ('SARIMA','Ensemble') else '--'
            lw = 2.2 if mname in ('SARIMA','Ensemble') else 1.4
            ax_line.plot(x, vals, color=mc, linestyle=ls,
                         linewidth=lw, marker='o', markersize=3.5,
                         label=mname, alpha=0.88)

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(districts, fontsize=6.5, rotation=55, ha='right')
    ax_line.set_title(f'{commodity} — 2026 Annual Avg Forecast by District & Model',
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

# =============================================================================
# Apply: insert three new cells before the Appendix (cell 43)
# =============================================================================
cells = nb['cells']

# Find appendix
appendix_idx = None
for i in range(len(cells)-1, -1, -1):
    if cells[i]['cell_type'] == 'markdown':
        appendix_idx = i
        break

print(f"Inserting before cell {appendix_idx} (Appendix)")

# Insert in reverse order to preserve indices
cells.insert(appendix_idx, code_cell(FIG17))
cells.insert(appendix_idx, code_cell(FIG16))
cells.insert(appendix_idx, code_cell(FIG15))

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook saved. Total cells: {len(cells)}")
print(f"  Cell {appendix_idx}   — Figure 15: ML In-Sample Fit (actual vs SVR/RF/LSTM + residuals)")
print(f"  Cell {appendix_idx+1} — Figure 16: Seasonal Pattern Comparison (SPI line + heatmap per model)")
print(f"  Cell {appendix_idx+2} — Figure 17: District Price Spread (box plot + per-district line per model)")
