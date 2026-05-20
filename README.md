# Hyperliquid Traders × Bitcoin Fear & Greed Index

> **Sentiment-Driven Trading Behaviour Analysis**  
> *Does market mood actually affect how well crypto traders perform? This project answers that with 211,000 real trades and statistical proof.*

---

## 📌 Overview

This project is a **data analysis system** that merges two years of real Hyperliquid trading data (211,218 trades) with the daily Bitcoin Fear & Greed Index to quantitatively answer:

> *"Does the mood of the market — whether people are scared or greedy — affect trader performance?"*

The output includes **8 professional charts**, **statistical proofs**, **6 trader personality archetypes**, and a **ready-to-use trading bot strategy rule engine** — all generated automatically.

---

## 🗂️ Project Structure

```
├── analysis.py                          # Core analysis pipeline
├── build_notebook.py                    # Auto-generates the Jupyter Notebook
├── hyperliquid_sentiment_analysis.ipynb # Interactive report (Jupyter Notebook)
├── fear_greed_index.csv                 # Bitcoin Fear & Greed Index (daily, 2023–2025)
├── historical_data.csv                  # Hyperliquid trade history (211,224 rows)
├── eda_summary.csv                      # Mood-by-mood performance breakdown
├── trader_profiles.csv                  # Per-trader metrics and archetypes
├── archetype_performance.csv            # Archetype-level performance summary
└── charts/
    ├── chart1_winrate_pnl.png           # Win rate & avg profit by mood
    ├── chart2_volume_distribution.png   # BUY vs SELL volume by mood
    ├── chart3_pnl_boxplot.png           # Profit/loss distribution (box plot)
    ├── chart4_directional_bias.png      # Directional bias (BUY/SELL %) by mood
    ├── chart5_archetype_pnl.png         # Avg PnL per trader archetype
    ├── chart6_correlation.png           # Mood score vs profit scatter plot
    ├── chart7_cumulative_pnl_timeline.png # Cumulative PnL over time, coloured by mood
    └── chart8_cross_vs_isolated.png     # Cross-margin vs Isolated-margin comparison
```

---

## 📊 Key Findings

| Finding | Detail |
|---|---|
| 🏆 **Best mood to trade** | Extreme Greed — **46.5% win rate**, **$67.89 avg profit/trade** |
| 😨 **Worst mood to trade** | Extreme Fear — only **37.1% win rate** |
| 📉 **Leverage kills profits** | Cross-margin traders earn **$35.63/trade** vs **$68.58/trade** for isolated-margin (**92% gap**, p = 7.15×10⁻¹³) |
| 📈 **Mood correlation** | Pearson r = 0.0081 — mood is real but weak as a standalone predictor |
| 💡 **Fear Opportunists** | Avg total PnL of **$248,002** — second-highest archetype |
| ⚠️ **FOMO Traders** | Use cross-margin **92.8%** of the time — high leverage, low win rate (28.5%) |

---

## 📈 Charts Preview

| Chart | Description |
|---|---|
| Chart 1 | Win rate & average profit for each mood category |
| Chart 2 | BUY vs SELL trading volume per mood |
| Chart 3 | PnL distribution box plot (outlier-clipped to ±$500) |
| Chart 4 | Directional bias (% BUY vs SELL) per mood |
| Chart 5 | Average PnL per trader archetype |
| Chart 6 | Mood score vs profit scatter plot (6,000-trade sample) |
| Chart 7 | Cumulative PnL timeline coloured by daily mood |
| Chart 8 | Cross-margin vs Isolated-margin win rate & profit |

---

## 🧠 Trader Archetypes

Six behaviour-based personality types identified from the data:

| Archetype | Rule | Insight |
|---|---|---|
| **Contrarian Buyer** | >40% trades in Extreme Fear & >55% BUY | Buys when others panic-sell |
| **FOMO Trader** | >40% trades in Extreme Greed & >55% BUY | Buys at peak hype |
| **Fear Opportunist** | Avg mood < 35 & total profit > 0 | Thrives in fearful markets |
| **Greed Victim** | Avg mood > 55 & total profit < 0 | Loses money chasing the hype |
| **Consistent Winner** | Win rate > 55% | Beats the market regardless of mood |
| **Mixed Sentiment** | None of the above | No dominant mood pattern |

---

## 🤖 Trading Bot Strategy (Ready to Use)

```python
def sentiment_execution_filter(fg_class, signal_side, base_position_size):
    """
    Adjusts trade size based on current market mood.
    fg_class          = current Fear & Greed label (e.g., 'Extreme Greed')
    signal_side       = 'BUY' or 'SELL'
    base_position_size = your default trade size in USD
    """
    if fg_class == 'Extreme Fear':
        if signal_side == 'SELL':
            return 0                            # Skip — 63% chance of losing
        else:
            return base_position_size * 0.6    # Reduce size, noisy market

    elif fg_class == 'Fear':
        return base_position_size * 0.85

    elif fg_class == 'Neutral':
        return base_position_size * 1.0

    elif fg_class == 'Greed':
        return base_position_size * 1.15

    elif fg_class == 'Extreme Greed':
        if signal_side == 'BUY':
            return base_position_size * 1.40   # Best win rate — scale up
        else:
            return base_position_size * 0.80
```

---

## 🚀 How to Run

### Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `nbformat`

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy nbformat
```

### Step 1 — Run the analysis
```bash
python analysis.py
```
Generates all 8 charts in `charts/` and 3 summary CSV files.

### Step 2 — Build the notebook
```bash
python build_notebook.py
```
Creates `hyperliquid_sentiment_analysis.ipynb`.

### Step 3 — Open the notebook
```bash
jupyter notebook hyperliquid_sentiment_analysis.ipynb
```

> **Non-technical users:** Open the notebook directly via [Google Colab](https://colab.research.google.com) — no installation needed. Or simply browse the `charts/` folder for pre-generated visuals.

---

## 📦 Datasets

The datasets are included in this repository and also available via Google Drive:

| File | Description | Rows | Download |
|---|---|---|---|
| `historical_data.csv` | Hyperliquid trade-level data (accounts, PnL, margin type, timestamps) | 211,224 | [📥 Download](https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing) |
| `fear_greed_index.csv` | Daily Bitcoin Fear & Greed Index scores (0–100) with classification labels | ~730 | [📥 Download](https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data loading, merging, grouping, feature engineering |
| **NumPy** | Fast mathematical operations |
| **Matplotlib** | Chart generation (full dark-theme control) |
| **Seaborn** | Statistical styling and theme consistency |
| **SciPy** | Pearson correlation & Welch t-tests |
| **Jupyter / nbformat** | Interactive notebook report generation |

---

## 📐 Statistical Tests

| Test | Question | Result |
|---|---|---|
| Pearson Correlation | Does mood score predict profit linearly? | r = 0.0081, p = 0.00019 (real but weak) |
| Welch t-test | Is Extreme Greed profit > Extreme Fear profit? | t = −3.851, **p = 0.000118** ✅ |
| Welch t-test | Does cross-margin earn less than isolated-margin? | t = −7.177, **p = 7.15×10⁻¹³** ✅ |

---

## 🎯 Use Cases

- **Crypto platforms** → Live mood-based risk warnings for users
- **Algo trading bots** → Plug in the strategy rule engine above
- **Individual traders** → Identify your archetype and improve your strategy
- **Financial research** → Replicable template for sentiment-vs-performance studies
- **Risk management** → Data-backed justification for leverage limits

---

## 👤 Author

**Sonika**  
Primetrade.ai Internship Assignment — May 2026  
Analysis pipeline: `analysis.py` | Notebook builder: `build_notebook.py`
