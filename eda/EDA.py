# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: practicum
#     language: python
#     name: practicum
# ---

# %% [markdown]
# # EDA: Bitcoin + Polymarket — Technical Appendix
#
# This notebook is the **comprehensive EDA reference** supporting `EDA_Executive.ipynb`.
#
# ## Goals
# 1. Load, validate, and profile Coin Metrics (BTC on-chain) and Polymarket datasets
# 2. Perform rigorous data integrity and completeness checks
# 3. Explore temporal structure, distributions, and regime dynamics
# 4. Evaluate whether prediction-market signals show utility for BTC accumulation modeling
#
# ## Data Sources
# | Source | File | Description |
# |--------|------|-------------|
# | Coin Metrics | `coinmetrics_btc.csv` | Daily on-chain + market metrics (~50 columns) |
# | Polymarket | `finance_politics_markets.parquet` | Market metadata (questions, categories, volume) |
# | Polymarket | `finance_politics_odds_history.parquet` | Time-series price/probability snapshots |
# | Polymarket | `finance_politics_summary.parquet` | Per-market summary stats |
# | Polymarket | `finance_politics_trades.parquet` | Granular trade data |
# | Polymarket | `finance_politics_tokens.parquet` | Outcome token mappings |
# | Polymarket | `finance_politics_event_stats.parquet` | Event-level aggregates |

# %%

# %% [markdown]
# ---
# ## 0 · Setup & Configuration

# %%
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings, textwrap

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 150, 'figure.facecolor': 'white'})

# --- Path resolution ---
cwd = Path.cwd().resolve()
candidates = [cwd, cwd.parent, cwd.parent.parent]
PROJECT_ROOT = Path('/home/jovyan/work/bitcoin-analytics-capstone-template')
EDA_DIR      = PROJECT_ROOT / 'eda'
PLOTS_DIR    = EDA_DIR / 'plots'
DATA_DIR     = PROJECT_ROOT / 'data' 
COINMETRICS_PATH = DATA_DIR / 'Coin Metrics' / 'coinmetrics_btc.csv'
POLYMARKET_DIR   = DATA_DIR / 'Polymarket'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"{COINMETRICS_PATH}")
print(f'PROJECT_ROOT : {PROJECT_ROOT}')
print(f'BTC CSV exists: {COINMETRICS_PATH.exists()}')
print(f'Poly dir exists: {POLYMARKET_DIR.exists()}')

# %%
import sys
print(sys.executable)


# %%
# --- Memory tracking utilities ---
def get_memory_usage_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return float('nan')

def format_memory(mb: float) -> str:
    if np.isnan(mb): return 'N/A'
    return f'{mb/1024:.2f} GB' if mb >= 1024 else f'{mb:.2f} MB'

@contextmanager
def track_memory(operation_name: str):
    before = get_memory_usage_mb()
    print(f'[Mem] Before {operation_name}: {format_memory(before)}')
    try:
        yield
    finally:
        after = get_memory_usage_mb()
        delta = after - before if not (np.isnan(after) or np.isnan(before)) else float('nan')
        print(f'[Mem] After  {operation_name}: {format_memory(after)} (Δ {format_memory(delta)})')

# --- Datetime helpers ---
def safe_to_datetime(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8).str.to_datetime(strict=False)

def to_date_if_present(df: pl.DataFrame, candidates: list[str]) -> pl.DataFrame:
    existing = [c for c in candidates if c in df.columns]
    if not existing: return df
    return df.with_columns([safe_to_datetime(pl.col(c)).alias(c) for c in existing])


# %% [markdown]
# ---
# ## 0.1 · Data Loading

# %%
def load_bitcoin_data(filepath: Path) -> Optional[pl.DataFrame]:
    if not filepath.exists():
        print(f'⚠ Bitcoin file not found: {filepath}'); return None
    with track_memory('loading Bitcoin data'):
        df = (pl.scan_csv(filepath, infer_schema_length=10000)
              .with_columns(safe_to_datetime(pl.col('time')).alias('time'))
              .collect())
    print(f'✓ Loaded BTC rows: {len(df):,}  cols: {len(df.columns)}')
    return df


def load_polymarket_data(datadir: Path) -> dict[str, pl.DataFrame]:
    """Load all available Polymarket parquet files with timestamp correction."""
    files = {
        'markets':     datadir / 'finance_politics_markets.parquet',
        'odds':        datadir / 'finance_politics_odds_history.parquet',
        'summary':     datadir / 'finance_politics_summary.parquet',
        'tokens':      datadir / 'finance_politics_tokens.parquet',
        'trades':      datadir / 'finance_politics_trades.parquet',
        'event_stats': datadir / 'finance_politics_event_stats.parquet',
    }
    out: dict[str, pl.DataFrame] = {}
    datetime_candidates = ['timestamp','time','ts','created_at','updated_at','end_date','closed_at',
                           'first_trade','last_trade','first_market_start','last_market_end']
    for name, path in files.items():
        if not path.exists():
            print(f'⚠ Missing Polymarket file ({name}): {path.name}'); continue
        with track_memory(f'loading Polymarket {name}'):
            df = pl.scan_parquet(path).collect()
        present = [c for c in datetime_candidates if c in df.columns]
        if present:
            df = to_date_if_present(df, present)
        out[name] = df
        print(f'✓ {name}: {len(df):,} rows × {len(df.columns)} cols')
    return out


btc_df    = load_bitcoin_data(COINMETRICS_PATH)
poly_data = load_polymarket_data(POLYMARKET_DIR)

print('\n── Loaded objects ──')
print(f'btc_df is None: {btc_df is None}')
print(f'Polymarket tables: {list(poly_data.keys())}')

# %%
if btc_df is not None:
    print('── BTC preview ──')
    display(btc_df.head(3))

for name, df in poly_data.items():
    print(f'\n── {name} preview ──')
    display(df.head(3))


# %% [markdown]
# ---
# # 1 · General Dataset Overview
#
# We systematically check every loaded table for:
# - **Shape & schema** (column types, counts)
# - **Missingness** (null rates, patterns)
# - **Duplicates & monotonicity** (time-series integrity)
# - **Descriptive statistics** (central tendency, dispersion, tails)
# - **Distributions** (histograms + skewness/kurtosis for key columns)

# %%
def profile_dataframe(df: pl.DataFrame, name: str) -> None:
    """Comprehensive profiling: shape, types, nulls, duplicates."""
    print(f'\n{"═"*60}')
    print(f' {name.upper()}')
    print(f'{"═"*60}')
    print(f'Shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
    
    # Schema
    schema_df = pl.DataFrame({'column': list(df.schema.keys()),
                              'dtype': [str(v) for v in df.schema.values()]})
    display(schema_df)
    
    # Nulls
    nulls = (df.null_count()
             .transpose(include_header=True, header_name='column', column_names=['null_count'])
             .with_columns((pl.col('null_count') / len(df) * 100).round(2).alias('null_pct')))
    high_null = nulls.filter(pl.col('null_pct') > 0).sort('null_pct', descending=True)
    if len(high_null) > 0:
        print(f'\nColumns with nulls ({len(high_null)}):')
        display(high_null.head(20))
    else:
        print('\n✓ No null values in any column.')
    
    # Exact-row duplicates
    n_dup = len(df) - len(df.unique())
    print(f'\nExact duplicate rows: {n_dup:,} ({n_dup/max(len(df),1)*100:.2f}%)')
    
    # Numeric describe
    numeric_cols = [c for c, dt in df.schema.items() if dt.is_numeric()]
    if numeric_cols:
        print(f'\nDescriptive statistics ({len(numeric_cols)} numeric columns):')
        display(df.select(numeric_cols).describe())


# Profile all datasets
if btc_df is not None:
    profile_dataframe(btc_df, 'Coin Metrics BTC')

for name, df in poly_data.items():
    profile_dataframe(df, f'Polymarket · {name}')


# %% [markdown]
# ### 1.1 · Time-Series Integrity
#
# For time-indexed data, we verify:
# - Date ranges (min/max)
# - Whether timestamps are sorted (monotonicity)
# - Missing calendar days (gaps)

# %%
def check_time_integrity(df: pl.DataFrame, label: str, time_col: str = 'time') -> None:
    """Check monotonicity, gaps, and range for a time column."""
    if time_col not in df.columns:
        print(f'{label}: column "{time_col}" not found'); return
    
    ts = df[time_col].drop_nulls().sort()
    print(f'\n── {label} · {time_col} ──')
    print(f'  Range : {ts.min()} → {ts.max()}')
    print(f'  Count : {len(ts):,}  Nulls: {df[time_col].null_count():,}')
    
    # Monotonicity
    is_sorted = ts.equals(ts.sort())
    print(f'  Sorted: {"✓ Yes" if is_sorted else "✗ No"}')
    
    # Daily gap analysis (only for date-level granularity)
    try:
        dates = ts.dt.date().unique().sort()
        if len(dates) > 1:
            date_range = pl.date_range(dates.min(), dates.max(), '1d', eager=True)
            n_expected = len(date_range)
            n_actual   = len(dates)
            n_gaps     = n_expected - n_actual
            print(f'  Calendar days expected: {n_expected:,}  actual: {n_actual:,}  gaps: {n_gaps:,}')
            if 0 < n_gaps <= 10:
                missing = date_range.filter(~date_range.is_in(dates))
                print(f'  Missing dates: {missing.to_list()}')
    except Exception as e:
        print(f'  (gap analysis skipped: {e})')


if btc_df is not None:
    check_time_integrity(btc_df, 'BTC', 'time')

# Check key Polymarket time columns
for tbl, tcol in [('odds','timestamp'), ('trades','timestamp'),
                   ('markets','created_at'), ('summary','first_trade')]:
    if tbl in poly_data and tcol in poly_data[tbl].columns:
        check_time_integrity(poly_data[tbl], f'Poly·{tbl}', tcol)

# %% [markdown]
# ### 1.2 · Distribution Analysis
#
# Examine the shape of key numeric columns — skewness, kurtosis, and visual histograms.
# Understanding tail behavior is critical for accumulation strategy design.

# %%
if btc_df is not None:
    key_metrics = ['PriceUSD','CapMrktCurUSD','HashRate','TxCnt','FeeTotUSD',
                   'FlowInExUSD','FlowOutExUSD','VtyDayRet30d']
    available = [c for c in key_metrics if c in btc_df.columns]
    
    if available:
        n = len(available)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
        axes = np.atleast_1d(axes).flatten()
        
        for i, col in enumerate(available):
            vals = btc_df[col].drop_nulls().to_numpy()
            ax = axes[i]
            ax.hist(vals, bins=60, edgecolor='white', alpha=0.8)
            ax.set_title(col, fontsize=10)
            skew = float(np.nanmean(((vals - np.nanmean(vals)) / np.nanstd(vals))**3)) if len(vals) > 3 else 0
            ax.text(0.95, 0.95, f'skew={skew:.2f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8, color='gray')
        
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        fig.suptitle('BTC Metric Distributions', fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'btc_distributions.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Saved: {PLOTS_DIR / "btc_distributions.png"}')

# %% [markdown]
# ---
# # 2 · Bitcoin On-Chain Exploration
#
# Deep dive into BTC price dynamics, volatility regimes, and on-chain fundamentals.
# These features form the backbone of any accumulation model.

# %% [markdown]
# ### 2.1 · Price History & Volatility Regimes

# %%
if btc_df is not None and {'time','PriceUSD'}.issubset(set(btc_df.columns)):
    btc_ts = (btc_df.select('time','PriceUSD').drop_nulls().sort('time')
              .with_columns([
                  pl.col('PriceUSD').pct_change().alias('daily_ret'),
              ])
              .with_columns([
                  pl.col('daily_ret').rolling_std(window_size=30).alias('vol_30d'),
                  pl.col('daily_ret').rolling_mean(window_size=200).alias('ma200_ret'),
                  pl.col('PriceUSD').rolling_mean(window_size=200).alias('sma200'),
              ]))
    pdf = btc_ts.to_pandas()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.5, 1.5]})
    
    # Price + SMA200
    axes[0].semilogy(pdf['time'], pdf['PriceUSD'], label='BTC Price', linewidth=0.9)
    axes[0].semilogy(pdf['time'], pdf['sma200'], label='200-day SMA', linewidth=0.7, alpha=0.7)
    axes[0].set_ylabel('Price (USD, log scale)')
    axes[0].set_title('Bitcoin Price History with 200-Day Moving Average')
    axes[0].legend()
    
    # Daily returns
    axes[1].bar(pdf['time'], pdf['daily_ret'], width=1, alpha=0.5, color='steelblue')
    axes[1].set_ylabel('Daily Return')
    axes[1].set_title('Daily Returns')
    axes[1].axhline(0, color='black', linewidth=0.5)
    
    # 30d vol
    axes[2].fill_between(pdf['time'], 0, pdf['vol_30d'], alpha=0.4, color='coral')
    axes[2].set_ylabel('30D Vol (σ)')
    axes[2].set_title('30-Day Rolling Volatility')
    axes[2].set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'btc_price_vol_returns.png', dpi=150)
    plt.show()
    print(f'Saved: btc_price_vol_returns.png')
else:
    print('BTC price plot skipped: data not loaded.')

# %% [markdown]
# ### 2.2 · On-Chain Fundamentals (MVRV, NVT, Exchange Flows)
#
# These metrics are widely used signals in crypto-native valuation:
# - **MVRV** (Market Value / Realised Value): values > 3 historically indicate overheating
# - **NVT** (Network Value to Transactions): high NVT suggests price outpacing utility
# - **Exchange flows**: net outflows from exchanges often precede supply squeezes

# %%
if btc_df is not None:
    onchain_cols = {
        'CapMVRVCur': 'MVRV Ratio',
        'NVTAdj': 'NVT (Adjusted)',
        'HashRate': 'Hash Rate (TH/s)',
        'FlowInExUSD': 'Exchange Inflow (USD)',
        'FlowOutExUSD': 'Exchange Outflow (USD)',
        'SplyActPct1yr': 'Active Supply % (1yr)',
    }
    avail = {k: v for k, v in onchain_cols.items() if k in btc_df.columns}
    
    if avail and 'time' in btc_df.columns:
        n = len(avail)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3*n), sharex=True)
        if n == 1: axes = [axes]
        pdf = btc_df.select(['time'] + list(avail.keys())).drop_nulls(subset=['time']).sort('time').to_pandas()
        
        for ax, (col, label) in zip(axes, avail.items()):
            ax.plot(pdf['time'], pdf[col], linewidth=0.8)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(label, fontsize=10)
        
        axes[-1].set_xlabel('Date')
        fig.suptitle('BTC On-Chain Fundamentals Over Time', fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'btc_onchain_fundamentals.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print('On-chain metrics not available in loaded data.')

# %% [markdown]
# ### 2.3 · Correlation Structure of BTC Metrics
#
# Understanding co-movement helps identify redundant vs. complementary features for modeling.

# %%
if btc_df is not None:
    corr_candidates = ['PriceUSD','CapMrktCurUSD','CapMVRVCur','CapRealUSD',
                       'HashRate','TxCnt','NVTAdj','FlowInExUSD','FlowOutExUSD',
                       'VtyDayRet30d','SplyActPct1yr','FeeTotUSD']
    corr_cols = [c for c in corr_candidates if c in btc_df.columns]
    
    if len(corr_cols) >= 3:
        corr_pd = btc_df.select(corr_cols).to_pandas()
        corr_matrix = corr_pd.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title('Correlation Matrix — BTC On-Chain Metrics', fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'btc_correlation_matrix.png', dpi=150)
        plt.show()
        print('Key observations:')
        print('  • PriceUSD and CapMrktCurUSD are near-perfectly correlated (expected)')
        print('  • Look for low-correlation metrics as independent feature candidates')

# %% [markdown]
# ---
# # 3 · Polymarket Exploration
#
# Understanding the structure, activity, and signal quality of prediction-market data before
# attempting cross-feature analysis with BTC.

# %% [markdown]
# ### 3.1 · Market Landscape

# %%
markets_df = poly_data.get('markets')
if markets_df is not None:
    print(f'Total markets: {len(markets_df):,}')
    
    if 'active' in markets_df.columns:
        active = markets_df['active'].sum()
        print(f'Active: {active:,}  |  Closed: {len(markets_df) - active:,}')
    
    if 'volume' in markets_df.columns:
        vol_stats = markets_df['volume'].describe()
        print(f'\nVolume distribution:')
        display(vol_stats)
    
    # Category breakdown
    cat_col = next((c for c in ['category','category_slug'] if c in markets_df.columns), None)
    vol_col = next((c for c in ['volume','total_volume'] if c in markets_df.columns), None)
    
    if cat_col and vol_col:
        cat_summary = (markets_df.group_by(cat_col)
                       .agg([pl.len().alias('n_markets'),
                             pl.col(vol_col).sum().alias('total_vol'),
                             pl.col(vol_col).mean().alias('avg_vol')])
                       .sort('total_vol', descending=True))
        print(f'\nCategory breakdown:')
        display(cat_summary)
        
        # Bar chart
        top = cat_summary.head(12).to_pandas()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=top, x='total_vol', y=cat_col, ax=ax, palette='viridis')
        ax.set_title('Top Polymarket Categories by Total Volume')
        ax.set_xlabel('Total Volume (USD)')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'polymarket_categories.png', dpi=150)
        plt.show()

# %% [markdown]
# ### 3.2 · Temporal Activity (Odds & Trades)

# %%
odds_df = poly_data.get('odds')
trades_df = poly_data.get('trades')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Odds activity over time
ts_col = None
if odds_df is not None:
    ts_col = next((c for c in ['timestamp','time'] if c in odds_df.columns), None)
if odds_df is not None and ts_col:
    daily_odds = (odds_df.with_columns(pl.col(ts_col).dt.date().alias('date'))
                  .group_by('date').agg(pl.len().alias('n_obs'))
                  .sort('date').to_pandas())
    axes[0].fill_between(daily_odds['date'], 0, daily_odds['n_obs'], alpha=0.5)
    axes[0].set_title('Daily Odds Observations')
    axes[0].set_ylabel('Count')
else:
    axes[0].text(0.5, 0.5, 'Odds data not available', ha='center', va='center', transform=axes[0].transAxes)

# Trades activity over time
t_col = None
if trades_df is not None:
    t_col = next((c for c in ['timestamp','time'] if c in trades_df.columns), None)
if trades_df is not None and t_col:
    daily_trades = (trades_df.with_columns(pl.col(t_col).dt.date().alias('date'))
                    .group_by('date').agg(pl.len().alias('n_trades'))
                    .sort('date').to_pandas())
    axes[1].fill_between(daily_trades['date'], 0, daily_trades['n_trades'], alpha=0.5, color='coral')
    axes[1].set_title('Daily Trade Count')
    axes[1].set_ylabel('Count')
else:
    axes[1].text(0.5, 0.5, 'Trades data not available', ha='center', va='center', transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'polymarket_temporal_activity.png', dpi=150)
plt.show()


# %% [markdown]
# ---
# # 4 · Prediction Market × BTC: Cross-Feature Analysis
#
# **Required investigation** per the outline: evaluate Polymarket utility for BTC accumulation strategies.
#
# We analyze:
# 1. Daily feature alignment (BTC returns vs. prediction-market activity)
# 2. Lead/lag correlations (do market signals lead BTC moves?)
# 3. Regime-conditioned analysis (do signals behave differently in high/low vol?)
# 4. Statistical significance tests

# %% [markdown]
# ### 4.1 · Daily Feature Merge & Baseline Correlation

# %%
def first_present(cols, options):
    return next((c for c in options if c in cols), None)

merged = None  # will be set if merge succeeds

if btc_df is not None and poly_data.get('odds') is not None:
    odds = poly_data['odds']
    btc_time = first_present(btc_df.columns, ['time','date','timestamp'])
    odds_time = first_present(odds.columns, ['timestamp','time','created_at'])
    odds_prob = first_present(odds.columns, ['price','probability','yes_price','odds','mid'])
    
    if btc_time and odds_time and odds_prob and 'PriceUSD' in btc_df.columns:
        # BTC daily features
        btc_daily = (btc_df.with_columns(pl.col(btc_time).dt.date().alias('date'))
                     .group_by('date').agg([
                         pl.col('PriceUSD').last().alias('btc_close'),
                         pl.col('PriceUSD').pct_change().last().alias('btc_ret'),
                     ]).sort('date'))
        
        # Add vol if available
        if 'VtyDayRet30d' in btc_df.columns:
            btc_vol = (btc_df.with_columns(pl.col(btc_time).dt.date().alias('date'))
                       .group_by('date').agg(pl.col('VtyDayRet30d').last().alias('btc_vol30d'))
                       .sort('date'))
            btc_daily = btc_daily.join(btc_vol, on='date', how='left')
        
        # Polymarket daily features
        odds_daily = (odds.with_columns(pl.col(odds_time).dt.date().alias('date'))
                      .group_by('date').agg([
                          pl.len().alias('pm_obs_count'),
                          pl.col(odds_prob).mean().alias('pm_avg_prob'),
                          pl.col(odds_prob).std().alias('pm_prob_dispersion'),
                          pl.col('market_id').n_unique().alias('pm_active_markets') if 'market_id' in odds.columns else pl.lit(None).alias('pm_active_markets'),
                      ]).sort('date'))
        
        merged = btc_daily.join(odds_daily, on='date', how='inner').drop_nulls()
        print(f'Merged daily rows: {len(merged):,}')
        print(f'Date range: {merged["date"].min()} → {merged["date"].max()}')
        display(merged.head())
        
        # Correlation heatmap
        if len(merged) > 10:
            feat_cols = [c for c in merged.columns if c != 'date']
            corr_pd = merged.select(feat_cols).to_pandas()
            corr = corr_pd.corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
                        center=0, square=True, ax=ax)
            ax.set_title('BTC Returns vs Polymarket Daily Features')
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'btc_vs_polymarket_corr.png', dpi=150)
            plt.show()
    else:
        print(f'Cross-feature analysis skipped: missing columns.')
else:
    print('Cross-feature analysis skipped: BTC or odds table not loaded.')

# %% [markdown]
# ### 4.2 · Lead/Lag Analysis
#
# Do Polymarket features at time $t-k$ predict BTC returns at time $t$?
# We test lags from 1 to 14 days.

# %%
if merged is not None and len(merged) > 30:
    pm_features = [c for c in merged.columns if c.startswith('pm_')]
    lags = list(range(1, 15))
    
    lag_corrs = {}
    merged_pd = merged.sort('date').to_pandas().set_index('date')
    
    for feat in pm_features:
        corrs = []
        for lag in lags:
            c = merged_pd[feat].shift(lag).corr(merged_pd['btc_ret'])
            corrs.append(c)
        lag_corrs[feat] = corrs
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for feat, corrs in lag_corrs.items():
        ax.plot(lags, corrs, marker='o', markersize=4, label=feat)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Correlation with BTC daily return')
    ax.set_title('Lead/Lag Correlation: Polymarket Features → BTC Return')
    ax.legend(fontsize=8, loc='best')
    ax.set_xticks(lags)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'leadlag_correlation.png', dpi=150)
    plt.show()
    
    # Print strongest signals
    print('\nStrongest lead signals (|corr| at any lag):')
    for feat, corrs in lag_corrs.items():
        best_idx = int(np.argmax(np.abs(corrs)))
        print(f'  {feat}: lag={lags[best_idx]}d, corr={corrs[best_idx]:.4f}')
else:
    print('Lead/lag analysis skipped: insufficient merged data.')

# %% [markdown]
# ### 4.3 · Regime-Conditioned Analysis
#
# Prediction-market signals may only be useful during specific volatility regimes.
# We split the data into high-vol and low-vol periods and re-examine correlations.

# %%
if merged is not None and 'btc_vol30d' in merged.columns and len(merged) > 30:
    vol_median = merged['btc_vol30d'].median()
    high_vol = merged.filter(pl.col('btc_vol30d') >= vol_median)
    low_vol  = merged.filter(pl.col('btc_vol30d') < vol_median)
    
    feat_cols = [c for c in merged.columns if c not in ('date','btc_vol30d')]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, subset, title in [(axes[0], high_vol, 'High Volatility'),
                               (axes[1], low_vol, 'Low Volatility')]:
        if len(subset) > 10:
            corr = subset.select(feat_cols).to_pandas().corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, square=True, ax=ax, cbar=False)
        ax.set_title(f'{title} (n={len(subset):,})')
    
    plt.suptitle('Correlation by Volatility Regime', fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'regime_correlation.png', dpi=150)
    plt.show()
elif merged is not None:
    print('Regime analysis: btc_vol30d not available; skipping.')
else:
    print('Regime analysis skipped: no merged data.')

# %% [markdown]
# ### 4.4 · Statistical Significance Tests
#
# We apply formal tests to check whether observed relationships are statistically meaningful:
# - **Augmented Dickey-Fuller** — stationarity of each series
# - **Spearman rank correlation** — non-parametric relationship test
# - **Granger causality** (if statsmodels available) — does PM data "Granger-cause" BTC returns?

# %%
from scipy import stats as sp_stats

if merged is not None and len(merged) > 30:
    merged_pd = merged.sort('date').to_pandas().set_index('date')
    pm_features = [c for c in merged_pd.columns if c.startswith('pm_')]
    
    # Spearman correlations with p-values
    print('── Spearman Rank Correlations (PM features vs btc_ret) ──')
    for feat in pm_features:
        clean = merged_pd[['btc_ret', feat]].dropna()
        if len(clean) > 10:
            rho, pval = sp_stats.spearmanr(clean['btc_ret'], clean[feat])
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            print(f'  {feat:30s}  ρ={rho:+.4f}  p={pval:.4f} {sig}')
    
    # ADF stationarity test
    print('\n── Augmented Dickey-Fuller (stationarity) ──')
    try:
        from statsmodels.tsa.stattools import adfuller
        for col in ['btc_ret'] + pm_features:
            series = merged_pd[col].dropna()
            if len(series) > 20:
                result = adfuller(series, maxlag=14, autolag='AIC')
                stationary = 'Yes' if result[1] < 0.05 else 'No'
                print(f'  {col:30s}  ADF={result[0]:.3f}  p={result[1]:.4f}  Stationary: {stationary}')
    except ImportError:
        print('  (statsmodels not installed; ADF test skipped)')
    
    # Granger causality
    print('\n── Granger Causality (PM → BTC return, max lag=7) ──')
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        for feat in pm_features:
            clean = merged_pd[['btc_ret', feat]].dropna()
            if len(clean) > 30:
                print(f'\n  Testing: {feat} → btc_ret')
                try:
                    result = grangercausalitytests(clean[['btc_ret', feat]], maxlag=7, verbose=False)
                    for lag, res in result.items():
                        f_pval = res[0]['ssr_ftest'][1]
                        sig = '***' if f_pval < 0.001 else '**' if f_pval < 0.01 else '*' if f_pval < 0.05 else ''
                        if f_pval < 0.1:
                            print(f'    lag={lag}: F-test p={f_pval:.4f} {sig}')
                except Exception as e:
                    print(f'    Error: {e}')
    except ImportError:
        print('  (statsmodels not installed; Granger test skipped)')
else:
    print('Statistical tests skipped: insufficient merged data.')

# %% [markdown]
# ### 4.5 · Scatter Plots: BTC Return vs Polymarket Features

# %%
if merged is not None and len(merged) > 10:
    pm_features = [c for c in merged.columns if c.startswith('pm_')]
    n = len(pm_features)
    if n > 0:
        fig, axes = plt.subplots(1, min(n, 4), figsize=(4.5*min(n,4), 4))
        if n == 1: axes = [axes]
        mpd = merged.to_pandas()
        for i, feat in enumerate(pm_features[:4]):
            axes[i].scatter(mpd[feat], mpd['btc_ret'], alpha=0.3, s=8)
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel('btc_ret')
            axes[i].axhline(0, color='gray', lw=0.5)
        plt.suptitle('BTC Return vs Polymarket Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'btc_vs_pm_scatter.png', dpi=150)
        plt.show()

# %% [markdown]
# ---
# # 5 · Formal Outcome
#
# After completing the cross-feature investigation above, we state one of the following:
#
# ### Outcome A: Could not discover use cases
# - We investigated cross-feature relationships between Polymarket activity and BTC market state.
# - We did not find stable, actionable signals for improving accumulation policy.
# - Key constraints: limited temporal overlap, noisy probabilities, or weak lead/lag behavior.
#
# ### Outcome B: Discovered interesting use cases
# - We identified candidate features from prediction-market dynamics that may improve accumulation decisions.
# - Examples: market participation shocks, probability-dispersion regimes, event clustering around macro narratives.
# - Next step: validate via out-of-sample tests in the backtesting pipeline.
#
# > **TODO**: After running all cells with real data, update this section with the actual conclusion,
# > citing specific correlation values, p-values, and lag results from sections 4.1–4.4.

# %% [markdown]
# ---
# # 6 · Next Steps & Extensions
#
# This EDA motivates several directions for follow-up modeling:
#
# 1. **Feature engineering**: Construct composite signals from MVRV + prediction-market dispersion
# 2. **Event windows**: Build event studies around major political/macro contract resolutions
# 3. **Regime detection**: Use hidden Markov models or changepoint detection for vol regimes
# 4. **Rolling stability**: Test whether lead/lag signals are stable across train/validation eras
# 5. **External data**: Integrate macro indicators (Fed funds rate, CPI) for richer context
# 6. **Accumulation backtests**: Feed strongest candidate features into the backtest pipeline
#    (see `template/model_development_template.py`)

# %%
print(f'\n[Mem] Final memory: {format_memory(get_memory_usage_mb())}')
print('\n✓ EDA notebook complete. Review plots/ directory for all saved figures.')

# %%

# %%
