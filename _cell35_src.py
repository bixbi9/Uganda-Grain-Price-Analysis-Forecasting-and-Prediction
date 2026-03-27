# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: FINAL SUMMARY DASHBOARD — ACTUALS + 2026 FORECAST
# ══════════════════════════════════════════════════════════════════════════════
n_comm = len(COMMODITIES)
fig, axes = plt.subplots(2, n_comm, figsize=(n_comm * 5, 12))
fig.patch.set_facecolor('#0A1520')
fig.suptitle('Uganda Grain Intelligence — 2024-2025 Actuals + 2026 SARIMAX Forecast',
             fontsize=16, fontweight='bold', color='white', y=0.98)

fd = pd.date_range(start='2026-01-01', periods=12, freq='MS')
month_labels = ['J','F','M','A','M','J','J','A','S','O','N','D']

for ci, commodity in enumerate(COMMODITIES):
    col = PALETTE.get(commodity, '#FFFFFF')

    # Valid districts for this commodity in the forecast dicts
    valid_dists = [d for d in COMMODITY_DISTRICTS[commodity]
                   if d in sarima_forecasts.get(commodity, {})
                   and d in sarimax_forecasts.get(commodity, {})]

    ax_top = axes[0, ci]
    ax_bot = axes[1, ci]
    ax_top.set_facecolor('#0F1923')
    ax_bot.set_facecolor('#0F1923')

    # ── Top: full timeline + forecast ribbon ──────────────────────────────
    nat_hist = (actuals[actuals['Commodity'] == commodity]
                .groupby('Date')['Price_UGX'].mean()
                .resample('MS').mean())
    nat_hist.index = pd.to_datetime(nat_hist.index)

    ax_top.plot(nat_hist.index, nat_hist.values, color=col, linewidth=2.2,
                label='Historical', zorder=3)
    ax_top.fill_between(nat_hist.index, nat_hist.values, alpha=0.10, color=col)

    if valid_dists:
        all_fc = np.array([
            0.4 * sarima_forecasts[commodity][r]['forecast'] +
            0.6 * sarimax_forecasts[commodity][r]['forecast']
            for r in valid_dists
        ])
        nat_fc  = all_fc.mean(axis=0)
        all_lo  = np.array([sarima_forecasts[commodity][r]['lower'] for r in valid_dists])
        all_hi  = np.array([sarimax_forecasts[commodity][r]['upper'] for r in valid_dists])

        ax_top.plot(fd, nat_fc, color='#FFD700', linewidth=2.5, linestyle='--',
                    label='2026 Forecast', zorder=5, marker='o', markersize=3.5)
        ax_top.fill_between(fd, all_lo.min(0), all_hi.max(0),
                            alpha=0.18, color='#FFD700', label='95% CI')
        ax_top.fill_between(fd, all_fc.min(0), all_fc.max(0),
                            alpha=0.28, color='#FFD700')
        ax_top.axvline(fd[0], color='white', linewidth=1, linestyle=':', alpha=0.6)

    ax_top.set_title(f'{commodity.upper()}', fontsize=10, color=col, fontweight='bold')
    ax_top.set_ylabel('UGX/kg', fontsize=7)
    ax_top.legend(fontsize=6)
    ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax_top.grid(True, alpha=0.3)

    # ── Bottom: 2026 Seasonal Price Index ─────────────────────────────────
    if valid_dists:
        fc_ann_avg = nat_fc.mean()
        si_2026    = nat_fc / fc_ann_avg * 100 if fc_ann_avg > 0 else nat_fc

        bar_colors = ['#F44336' if v > 110 else '#FF8C42' if v > 105 else
                      '#4CAF50' if v < 95  else '#FFC107'  if v < 100 else '#2196F3'
                      for v in si_2026]
        bars = ax_bot.bar(range(12), si_2026, color=bar_colors, alpha=0.85,
                          edgecolor='white', linewidth=0.5, width=0.7)
        ax_bot.axhline(100, color='white',   linewidth=1.5, linestyle='--', alpha=0.7)
        ax_bot.axhline(110, color='#FF5722', linewidth=1,   linestyle=':',  alpha=0.5)
        ax_bot.axhline(90,  color='#4CAF50', linewidth=1,   linestyle=':',  alpha=0.5)

        for bar, si_val in zip(bars, si_2026):
            ax_bot.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, f'{si_val:.1f}',
                        ha='center', va='bottom', fontsize=7,
                        fontweight='bold', color='white')
    else:
        ax_bot.text(0.5, 0.5, 'No Data', transform=ax_bot.transAxes,
                    ha='center', va='center', color='#555555', fontsize=11)

    ax_bot.set_xticks(range(12))
    ax_bot.set_xticklabels(month_labels, fontsize=8)
    ax_bot.set_title(f'{commodity} — SPI 2026', fontsize=9, color=col, fontweight='bold')
    ax_bot.set_ylabel('Index (100 = avg)', fontsize=7)
    ax_bot.set_ylim(75, 135)
    ax_bot.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('outputs/fig12_final_summary.png',
            dpi=150, bbox_inches='tight', facecolor='#0A1520')
plt.show()
plt.close()
print("Figure 12 saved: fig12_final_summary.png")
