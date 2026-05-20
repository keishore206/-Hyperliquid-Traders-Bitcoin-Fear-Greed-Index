
# ============================================================
#  Hyperliquid × Fear & Greed Index — Full Analysis Pipeline
# ============================================================

import warnings, os
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── colour palette ────────────────────────────────────────────
SENTIMENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
SENTIMENT_COLORS = {
    'Extreme Fear':  '#E74C3C',
    'Fear':          '#E67E22',
    'Neutral':       '#95A5A6',
    'Greed':         '#27AE60',
    'Extreme Greed': '#1A8F3C',
}
sns.set_theme(style='darkgrid', palette='muted', font='DejaVu Sans')
plt.rcParams.update({
    'figure.facecolor': '#0D1117',
    'axes.facecolor':   '#161B22',
    'axes.edgecolor':   '#30363D',
    'axes.labelcolor':  '#C9D1D9',
    'xtick.color':      '#C9D1D9',
    'ytick.color':      '#C9D1D9',
    'text.color':       '#C9D1D9',
    'grid.color':       '#21262D',
    'grid.linewidth':   0.6,
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
})

OUTPUT_DIR = 'charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD & MERGE
# ─────────────────────────────────────────────────────────────
print("Loading data ...")
hist  = pd.read_csv('historical_data.csv')
fg    = pd.read_csv('fear_greed_index.csv')

hist['trade_date'] = pd.to_datetime(hist['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
fg['trade_date']   = pd.to_datetime(fg['date']).dt.date

merged = pd.merge(hist, fg[['trade_date', 'value', 'classification']],
                  on='trade_date', how='inner')
merged['classification'] = pd.Categorical(merged['classification'],
                                          categories=SENTIMENT_ORDER, ordered=True)
merged['is_win']  = merged['Closed PnL'] > 0
merged['is_loss'] = merged['Closed PnL'] < 0

print(f"  Historical rows : {len(hist):,}")
print(f"  Merged rows     : {len(merged):,}")
print(f"  Date range      : {merged['trade_date'].min()} to {merged['trade_date'].max()}")
print()

# ─────────────────────────────────────────────────────────────
# 2. EDA — Win Rate, Volume, Directional Bias
# ─────────────────────────────────────────────────────────────
print("Running EDA ...")

eda = merged.groupby('classification', observed=True).agg(
    trade_count  = ('Closed PnL', 'count'),
    win_rate     = ('is_win',     'mean'),
    avg_pnl      = ('Closed PnL', 'mean'),
    median_pnl   = ('Closed PnL', 'median'),
    total_volume = ('Size USD',   'sum'),
    avg_volume   = ('Size USD',   'mean'),
    buy_ratio    = ('Side',       lambda x: (x == 'BUY').mean()),
).reset_index()
eda['win_rate_pct'] = eda['win_rate'] * 100
eda['buy_ratio_pct'] = eda['buy_ratio'] * 100

print(eda[['classification','trade_count','win_rate_pct','avg_pnl','avg_volume','buy_ratio_pct']].to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING — Trader-level profiles
# ─────────────────────────────────────────────────────────────
print("Engineering trader-level features ...")

trader = merged.groupby('Account').agg(
    total_trades      = ('Closed PnL', 'count'),
    total_pnl         = ('Closed PnL', 'sum'),
    avg_trade_size    = ('Size USD',   'mean'),
    win_rate          = ('is_win',     'mean'),
    extreme_fear_pct  = ('classification', lambda x: (x == 'Extreme Fear').mean()),
    extreme_greed_pct = ('classification', lambda x: (x == 'Extreme Greed').mean()),
    buy_ratio         = ('Side', lambda x: (x == 'BUY').mean()),
    avg_sentiment     = ('value', 'mean'),
    cross_ratio       = ('Crossed', 'mean'),    # fraction of trades using cross-margin
).reset_index()
trader = trader[trader['total_trades'] >= 10]   # at least 10 trades for reliability

# Archetype classification
def classify_archetype(row):
    if row['extreme_fear_pct'] > 0.40 and row['buy_ratio'] > 0.55:
        return 'Contrarian Buyer'
    elif row['extreme_greed_pct'] > 0.40 and row['buy_ratio'] > 0.55:
        return 'FOMO Trader'
    elif row['avg_sentiment'] < 35 and row['total_pnl'] > 0:
        return 'Fear Opportunist'
    elif row['avg_sentiment'] > 55 and row['total_pnl'] < 0:   # loosened: was >65
        return 'Greed Victim'
    elif row['win_rate'] > 0.55:
        return 'Consistent Winner'
    else:
        return 'Mixed Sentiment'

trader['archetype'] = trader.apply(classify_archetype, axis=1)

archetype_perf = trader.groupby('archetype').agg(
    count      = ('Account', 'count'),
    avg_pnl    = ('total_pnl', 'mean'),
    med_pnl    = ('total_pnl', 'median'),
    win_rate   = ('win_rate', 'mean'),
    avg_size   = ('avg_trade_size', 'mean'),
    cross_ratio= ('cross_ratio', 'mean'),
).reset_index().sort_values('avg_pnl', ascending=False)

print("Archetype Performance:")
print(archetype_perf.to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. STATISTICAL TESTING
# ─────────────────────────────────────────────────────────────────────────────
print("Statistical analysis ...")

corr, pval = stats.pearsonr(merged['value'].dropna(), merged['Closed PnL'].dropna())
print(f"  Pearson r (sentiment vs PnL): {corr:.4f}  (p={pval:.4e})")

ef_pnl = merged[merged['classification'] == 'Extreme Fear']['Closed PnL']
eg_pnl = merged[merged['classification'] == 'Extreme Greed']['Closed PnL']
t_stat, t_p = stats.ttest_ind(ef_pnl, eg_pnl, equal_var=False)
print(f"  Welch t-test EF vs EG PnL: t={t_stat:.3f}, p={t_p:.4e}")

# ── Cross-margin vs Isolated-margin analysis ──────────────────────────────
cross_pnl    = merged[merged['Crossed'] == True]['Closed PnL']
isolated_pnl = merged[merged['Crossed'] == False]['Closed PnL']
tc, tp = stats.ttest_ind(cross_pnl, isolated_pnl, equal_var=False)
print(f"  Cross-margin avg PnL    : ${cross_pnl.mean():.2f}  (n={len(cross_pnl):,})")
print(f"  Isolated-margin avg PnL : ${isolated_pnl.mean():.2f}  (n={len(isolated_pnl):,})")
print(f"  Welch t-test (cross vs isolated): t={tc:.3f}, p={tp:.4e}")

# ─────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
print("\nGenerating charts ...")

# ── helper ──
def styled_bar(ax, x, y, colors, xlabel, ylabel, title):
    bars = ax.bar(x, y, color=colors, width=0.55, edgecolor='none', zorder=3)
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=15, ha='right', fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + abs(h)*0.02,
                f'{h:,.1f}', ha='center', va='bottom', fontsize=8.5, color='#C9D1D9')
    return bars

# ── CHART 1: Win Rate & Avg PnL side-by-side ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0D1117')
fig.suptitle('Win Rate & Average PnL by Market Sentiment', fontsize=15, color='#C9D1D9', y=1.01)

cats   = eda['classification'].tolist()
colors = [SENTIMENT_COLORS[c] for c in cats]

styled_bar(axes[0], cats, eda['win_rate_pct'], colors,
           'Sentiment', 'Win Rate (%)', 'Win Rate by Sentiment')
axes[0].axhline(50, color='#58A6FF', linewidth=1.2, linestyle='--', label='50% line', zorder=4)
axes[0].legend(fontsize=9)

styled_bar(axes[1], cats, eda['avg_pnl'], colors,
           'Sentiment', 'Avg Closed PnL (USD)', 'Average PnL by Sentiment')
axes[1].axhline(0, color='#58A6FF', linewidth=1.2, linestyle='--', zorder=4)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart1_winrate_pnl.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("  [OK] chart1_winrate_pnl.png")

# ── CHART 2: Trade Volume by Sentiment ─────────────────────
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0D1117')
ax.set_facecolor('#161B22')
vols_b = merged[merged['Side']=='BUY'].groupby('classification', observed=True)['Size USD'].sum() / 1e6
vols_s = merged[merged['Side']=='SELL'].groupby('classification', observed=True)['Size USD'].sum() / 1e6
x_pos = np.arange(len(SENTIMENT_ORDER))
width = 0.35
b1 = ax.bar(x_pos - width/2, [vols_b.get(c, 0) for c in SENTIMENT_ORDER],
            width, label='BUY', color='#2ECC71', edgecolor='none', zorder=3)
b2 = ax.bar(x_pos + width/2, [vols_s.get(c, 0) for c in SENTIMENT_ORDER],
            width, label='SELL', color='#E74C3C', edgecolor='none', zorder=3)
ax.set_xticks(x_pos)
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, ha='right', fontsize=9)
ax.set_xlabel('Sentiment Classification', labelpad=8)
ax.set_ylabel('Total Volume (Millions USD)', labelpad=8)
ax.set_title('Trade Volume Distribution by Sentiment & Direction', pad=10)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart2_volume_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("  [OK] chart2_volume_distribution.png")

# ── CHART 3: Boxplot — PnL per Sentiment ───────────────────
fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0D1117')
ax.set_facecolor('#161B22')
box_data = [merged[merged['classification']==c]['Closed PnL'].clip(-500, 500).dropna()
            for c in SENTIMENT_ORDER]
bp = ax.boxplot(box_data, patch_artist=True, notch=True,
                medianprops=dict(color='#F1C40F', linewidth=2),
                whiskerprops=dict(color='#C9D1D9'),
                capprops=dict(color='#C9D1D9'),
                flierprops=dict(marker='o', color='#58A6FF', alpha=0.2, markersize=3))
for patch, color in zip(bp['boxes'], [SENTIMENT_COLORS[c] for c in SENTIMENT_ORDER]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, ha='right', fontsize=9)
ax.set_xlabel('Sentiment Classification', labelpad=8)
ax.set_ylabel('Closed PnL (USD) — clipped ±$500', labelpad=8)
ax.set_title('PnL Distribution per Sentiment (Boxplot)', pad=10)
ax.axhline(0, color='#58A6FF', linewidth=1.2, linestyle='--', zorder=4)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart3_pnl_boxplot.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("  [OK] chart3_pnl_boxplot.png")

# ── CHART 4: Buy Ratio by Sentiment ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0D1117')
ax.set_facecolor('#161B22')
buy_vals = eda['buy_ratio_pct'].tolist()
sell_vals = [100-v for v in buy_vals]
x_pos = np.arange(len(cats))
ax.bar(x_pos, buy_vals,  label='BUY %',  color='#2ECC71', edgecolor='none', zorder=3)
ax.bar(x_pos, sell_vals, bottom=buy_vals, label='SELL %', color='#E74C3C', edgecolor='none', zorder=3)
ax.axhline(50, color='#F1C40F', linewidth=1.2, linestyle='--', zorder=4)
ax.set_xticks(x_pos)
ax.set_xticklabels(cats, rotation=15, ha='right', fontsize=9)
ax.set_xlabel('Sentiment Classification', labelpad=8)
ax.set_ylabel('Direction Mix (%)', labelpad=8)
ax.set_title('Directional Bias (BUY vs SELL) by Sentiment', pad=10)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart4_directional_bias.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("  [OK] chart4_directional_bias.png")

# ── CHART 5: Trader Archetype Profitability ─────────────────
fig, ax = plt.subplots(figsize=(11, 6), facecolor='#0D1117')
ax.set_facecolor('#161B22')
arch_colors = ['#1A8F3C' if v >= 0 else '#E74C3C' for v in archetype_perf['avg_pnl']]
bars = ax.barh(archetype_perf['archetype'], archetype_perf['avg_pnl'],
               color=arch_colors, edgecolor='none', zorder=3)
for bar in bars:
    w = bar.get_width()
    ax.text(w + (1 if w >= 0 else -1), bar.get_y() + bar.get_height()/2,
            f'${w:,.0f}', va='center', ha='left' if w >= 0 else 'right', fontsize=9, color='#C9D1D9')
ax.axvline(0, color='#C9D1D9', linewidth=0.8, linestyle='-')
ax.set_xlabel('Average Total PnL (USD)', labelpad=8)
ax.set_title('Average PnL by Trader Archetype', pad=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart5_archetype_pnl.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("  [OK] chart5_archetype_pnl.png")

# ── CHART 6: Correlation scatter (sampled) ─────────────────
sample = merged[merged['Closed PnL'].between(-300, 300)].sample(
    min(5000, len(merged)), random_state=42)
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0D1117')
ax.set_facecolor('#161B22')
scatter_colors = [SENTIMENT_COLORS[c] for c in sample['classification']]
ax.scatter(sample['value'], sample['Closed PnL'], c=scatter_colors,
           alpha=0.18, s=12, edgecolors='none')
m, b, r_val, *_ = stats.linregress(sample['value'], sample['Closed PnL'])
x_line = np.linspace(0, 100, 200)
ax.plot(x_line, m*x_line + b, color='#58A6FF', linewidth=2, label=f'Trend (r={r_val:.3f})')
legend_patches = [mpatches.Patch(color=v, label=k) for k,v in SENTIMENT_COLORS.items()]
ax.legend(handles=legend_patches + [ax.get_lines()[0]], fontsize=8, loc='upper left')
ax.set_xlabel('Fear & Greed Value (0-100)', labelpad=8)
ax.set_ylabel('Closed PnL (USD) — clipped ±$300', labelpad=8)
ax.set_title(f'Correlation: Market Sentiment vs Trade PnL  (r = {corr:.4f})', pad=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart6_correlation.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("  [OK] chart6_correlation.png")

# ── CHART 7: Cumulative PnL over time coloured by sentiment ────────────────
merged_sorted = merged.sort_values('trade_date').copy()
merged_sorted['cum_pnl'] = merged_sorted['Closed PnL'].cumsum() / 1e6   # in millions

daily = merged_sorted.groupby('trade_date').agg(
    cum_pnl   = ('Closed PnL', 'sum'),
    sentiment = ('classification', lambda x: x.mode()[0]),
).reset_index()
daily['cum_pnl'] = daily['cum_pnl'].cumsum() / 1e6
daily['trade_date'] = pd.to_datetime(daily['trade_date'])

fig, ax = plt.subplots(figsize=(14, 6), facecolor='#0D1117')
ax.set_facecolor('#161B22')

for i in range(len(daily) - 1):
    seg_color = SENTIMENT_COLORS.get(daily['sentiment'].iloc[i], '#95A5A6')
    ax.plot(daily['trade_date'].iloc[i:i+2],
            daily['cum_pnl'].iloc[i:i+2],
            color=seg_color, linewidth=1.8, solid_capstyle='round')

ax.fill_between(daily['trade_date'], daily['cum_pnl'],
                alpha=0.08, color='#58A6FF')
ax.axhline(0, color='#C9D1D9', linewidth=0.7, linestyle=':')
legend_patches = [mpatches.Patch(color=v, label=k) for k,v in SENTIMENT_COLORS.items()]
ax.legend(handles=legend_patches, fontsize=8.5, loc='upper left', framealpha=0.3)
ax.set_xlabel('Date', labelpad=8)
ax.set_ylabel('Cumulative PnL (Millions USD)', labelpad=8)
ax.set_title('Cumulative PnL Over Time — Coloured by Prevailing Market Sentiment', pad=10)
plt.xticks(rotation=20, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart7_cumulative_pnl_timeline.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("  [OK] chart7_cumulative_pnl_timeline.png")

# ── CHART 8: Cross-margin vs Isolated — Win Rate & Avg PnL by Sentiment ────
cross_eda    = merged[merged['Crossed']==True].groupby('classification', observed=True).agg(
    win_rate=('is_win','mean'), avg_pnl=('Closed PnL','mean')).reset_index()
isolated_eda = merged[merged['Crossed']==False].groupby('classification', observed=True).agg(
    win_rate=('is_win','mean'), avg_pnl=('Closed PnL','mean')).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#0D1117')
fig.suptitle('Cross-Margin vs Isolated-Margin: Win Rate & Avg PnL by Sentiment',
             fontsize=14, color='#C9D1D9', y=1.01)

x   = np.arange(len(SENTIMENT_ORDER))
w   = 0.35
for ax, metric, ylabel, title in zip(
        axes,
        ['win_rate', 'avg_pnl'],
        ['Win Rate (%)', 'Avg Closed PnL (USD)'],
        ['Win Rate: Cross vs Isolated Margin', 'Avg PnL: Cross vs Isolated Margin']):
    ax.set_facecolor('#161B22')
    cv = [cross_eda.set_index('classification')[metric].get(c, 0) for c in SENTIMENT_ORDER]
    iv = [isolated_eda.set_index('classification')[metric].get(c, 0) for c in SENTIMENT_ORDER]
    if metric == 'win_rate':
        cv = [v*100 for v in cv]
        iv = [v*100 for v in iv]
    ax.bar(x - w/2, cv, w, label='Cross-margin',    color='#9B59B6', edgecolor='none', zorder=3)
    ax.bar(x + w/2, iv, w, label='Isolated-margin', color='#3498DB', edgecolor='none', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, pad=10)
    ax.legend(fontsize=9)
    if metric == 'win_rate':
        ax.axhline(50, color='#F1C40F', linewidth=1.2, linestyle='--', zorder=4)
    else:
        ax.axhline(0, color='#58A6FF', linewidth=1.2, linestyle='--', zorder=4)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart8_cross_vs_isolated.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("  [OK] chart8_cross_vs_isolated.png")

print("\nAll charts saved to ./charts/")
print("\nDone! Analysis complete.")

# ─────────────────────────────────────────────────────────────
# Save summary stats for notebook reference
# ─────────────────────────────────────────────────────────────
eda.to_csv('eda_summary.csv', index=False)
trader.to_csv('trader_profiles.csv', index=False)
archetype_perf.to_csv('archetype_performance.csv', index=False)
print("Summary CSVs written.")
