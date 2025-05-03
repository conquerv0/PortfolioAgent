import pandas as pd, numpy as np, yfinance as yf, matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
# ------------------------------------------------------------------
# 1.  CONFIG & DATA
# ------------------------------------------------------------------
from src.config.settings import PORTFOLIOS  
def load_bond_etf_returns(filepath: str) -> pd.DataFrame:
    """
    Load bond ETF data from a CSV file and compute the full daily return for each ETF.
    
    The CSV file is expected to have at least the following columns:
        - date: Trading date
        - ticker: ETF ticker
        - prc: Price (which may be negative for adjustments)
        - ret: Daily price return (as a decimal)
        - divamt: Dividend (or coupon) cash amount
        - vwretd: Value-weighted total return (includes distributions)
        
    The function computes:
        dividend_yield = divamt / abs(prc)   if prc is nonzero, else 0.
        full_return = vwretd (if available) else ret + dividend_yield.
    
    The final DataFrame is pivoted so that the row index is the date and the columns are the ticker names.
    
    Parameters:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Daily full return DataFrame with date as index and ticker as columns.
    """
    # Load the CSV file; ensure the date column is parsed as datetime.
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # Ensure the required columns are present.
    required_columns = ['date', 'ticker', 'prc', 'ret', 'divamt', 'vwretd']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    
    # Compute dividend yield when price is nonzero.
    df['dividend_yield'] = df.apply(lambda row: row['divamt'] / abs(row['prc']) if row['prc'] != 0 else 0, axis=1)
    
    # Compute full return:
    # If vwretd is available (non-null), use it; otherwise, add the dividend yield to the price return.
    df['computed_return'] = df.apply(
        lambda row: row['vwretd'] if pd.notnull(row['vwretd']) else (row['ret'] + row['dividend_yield']),
        axis=1
    )
    
    # Pivot the DataFrame so that 'date' becomes the index and each ticker is a column.
    full_return_df = df.pivot(index='date', columns='ticker', values='computed_return')
    full_return_df.fillna(0, inplace=True)
    
    return full_return_df

asset_class = "fx"  # fx, equity, fi, commodity
PRED_FILE   = f"data/predictions/{asset_class}_weekly_predictions.csv"  
PRICE_FILE  = f"data/features/{asset_class}_combined_features_weekly.csv"       # already saved
fx_tickers = [entry["etf"] for entry in PORTFOLIOS['fx'].get("currencies", [])]
fi_tickers = [entry["etf"] for entry in PORTFOLIOS['bond'].get("treasuries", [])]
equity_tickers = [entry["etf"] for entry in PORTFOLIOS["equity"].get("sectors", [])]
commodity_tickers = [entry["etf"] for entry in PORTFOLIOS["commodity"].get("sectors", [])]


if asset_class == "fx":
    TICKERS = fx_tickers
elif asset_class == "fi":
    TICKERS = fi_tickers
elif asset_class == "equity":
    TICKERS = equity_tickers
elif asset_class == "commodity":
    TICKERS = commodity_tickers
else:
    raise ValueError("Unsupported asset class")

# date range that matches back-test:
START, END = "2023-11-03", "2025-03-28"

# predictions  → wide matrix
pred = (pd.read_csv(PRED_FILE, parse_dates=["date"])
          .pivot(index="date", columns="etf", values="predicted_return")
          .loc[START:END, TICKERS]
          .astype(float))


# realised weekly return for NEXT week
if PRICE_FILE and os.path.exists(PRICE_FILE):
    px = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)[TICKERS]
else:                                                                        # fallback
    px = yf.download(TICKERS, start=START, end=END, auto_adjust=True)["Adj Close"]

if asset_class == "fi":
    daily = load_bond_etf_returns('data/bond_etf.csv')  # index=date, cols=tickers
    # restrict to our window
    daily = daily.loc[START:END, TICKERS]
    
    # 2) compound into weekly returns (Fri close to Fri close)
    weekly = (daily + 1.0).resample("W-FRI").prod().sub(1.0)
    
    # align with next‑week forecast
    # if pred_df is prediction for week t→t+1, we shift weekly returns up
    realised = weekly.shift(-1).reindex(pred.index).fillna(0)

# ------------------------------------------------------------------
# 2.  DIRECTIONAL & ERROR METRICS
# ------------------------------------------------------------------
directional_hit = (np.sign(pred) == np.sign(realised)).mean().mean()
common = pd.concat({'real': realised, 'pred': pred}, axis=1)

# keep only rows where both real & pred are present
mask = common['real'].notna() & common['pred'].notna()
real_vec = common['real'][mask].stack(dropna=True)
pred_vec = common['pred'][mask].stack(dropna=True)

mae = mean_absolute_error(real_vec, pred_vec)
mse = mean_squared_error(real_vec, pred_vec)
r2  = r2_score(real_vec,  pred_vec)

print(f"Directional accuracy : {directional_hit: .2%}")
print(f"Mean Abs Error (MAE) : {mae: .5f}")
print(f"Mean Sq  Error (MSE) : {mse: .5f}")
print(f"R²                   : {r2: .3f}")

# ------------------------------------------------------------------
# 3.  ACCURACY VS CONFIDENCE
# ------------------------------------------------------------------
# Calculate absolute prediction values as confidence measure
confidence = (pd.read_csv(PRED_FILE, parse_dates=["date"])
          .pivot(index="date", columns="etf", values="confidence")
          .loc[START:END, TICKERS]
          .astype(float))
common_conf = pd.concat({'real': realised, 'pred': pred, 'conf': confidence}, axis=1)

# Calculate accuracy for binned confidence levels
n_bins = 5
results = []

# Get the bin edges for creating labels later
all_confidence_values = []

for ticker in TICKERS:
    ticker_data = common_conf.xs(ticker, level=1, axis=1).dropna()
    if len(ticker_data) == 0:
        continue
        
    # Calculate directional accuracy (1 if correct direction, 0 if wrong)
    ticker_data['correct'] = (np.sign(ticker_data['pred']) == np.sign(ticker_data['real'])).astype(int)
    
    # Collect all confidence values for determining bin edges
    all_confidence_values.extend(ticker_data['conf'].values)
    
    # Bin by fixed confidence ranges instead of quantiles
    # Convert to percentage (0.50 to 0.59 becomes bin 0, 0.60 to 0.69 becomes bin 1, etc.)
    ticker_data['conf_bin'] = np.floor((ticker_data['conf'] * 100 - 50) / 10).clip(0, 4).astype(int)
    
    # Group by confidence bin and calculate mean accuracy
    binned_results = ticker_data.groupby('conf_bin').agg({
        'correct': 'mean',
        'conf': 'mean'
    }).reset_index()
    
    binned_results['ticker'] = ticker
    results.append(binned_results)

if results:
    # Combine results from all tickers
    all_results = pd.concat(results, ignore_index=True)
    
    # Calculate average accuracy by bin across all tickers
    avg_by_bin = all_results.groupby('conf_bin').agg({
        'correct': 'mean',
        'conf': 'mean'
    }).reset_index()
    
    # Sort by confidence bin
    avg_by_bin = avg_by_bin.sort_values('conf_bin')
    
    # Print debug information
    print("\nConfidence statistics:")
    print(f"Min confidence: {min(all_confidence_values):.4f}")
    print(f"Max confidence: {max(all_confidence_values):.4f}")
    print(f"Mean confidence: {np.mean(all_confidence_values):.4f}")
    
    # Fixed bin ranges (50-60%, 60-70%, etc.)
    fixed_ranges = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    
    # Count how many data points fall into each bin (but don't print)
    bin_counts = all_results.groupby('conf_bin').size()
    
    # ---------------- Graph 1: Accuracy by Confidence Bin ----------------
    plt.figure(figsize=(10, 8))
    
    # Create bar chart with confidence bin on x-axis
    plt.bar(avg_by_bin['conf_bin'], avg_by_bin['correct'], 
            width=0.7, color='#4A90E2', alpha=0.8)
    
    # Add gridlines
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', color='#E0E0E0')
    
    # Set axis limits and labels
    plt.xlabel('Confidence Range')
    plt.ylabel('Directional Accuracy')
    plt.title(f'Directional Accuracy by Confidence Range for {asset_class.upper()} ETFs')
    plt.ylim(0, 1)
    
    # Ensure all bins are represented in the plot
    all_bins = list(range(5))  # 0-4 for our 5 confidence ranges
    
    # If some bins have no data, we'll add them with zero height
    existing_bins = sorted(avg_by_bin['conf_bin'].unique())
    missing_bins = [b for b in all_bins if b not in existing_bins]
    
    # Add missing bins with zero height to the plot (silently)
    for bin_idx in missing_bins:
        plt.bar(bin_idx, 0, width=0.7, color='#4A90E2', alpha=0.3)  # lighter color for empty bins
    
    # Set x-ticks for all bins
    plt.xticks(all_bins, fixed_ranges)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'data/evaluation/{asset_class}/{asset_class}_accuracy_by_confidence_bin.png')
    
    # ---------------- Graph 2: Directional Accuracy by Ticker ----------------
    # Calculate average confidence and accuracy for each ticker
    ticker_stats = []
    
    for ticker in TICKERS:
        ticker_data = common_conf.xs(ticker, level=1, axis=1).dropna()
        if len(ticker_data) == 0:
            continue
            
        # Calculate directional accuracy
        dir_accuracy = (np.sign(ticker_data['pred']) == np.sign(ticker_data['real'])).mean()
        avg_confidence = ticker_data['conf'].mean()
        sample_size = len(ticker_data)
        
        ticker_stats.append({
            'Ticker': ticker,
            'Avg Confidence': avg_confidence,
            'Directional Accuracy': dir_accuracy,
            'Sample Size': sample_size
        })
    
    # Create DataFrame and sort by directional accuracy
    stats_df = pd.DataFrame(ticker_stats)
    
    plt.figure(figsize=(12, 8))
    
    # Set bar positions
    bar_positions = np.arange(len(stats_df))
    bar_width = 0.35
    
    # Create the plot with minimal blue bars and prominent orange bars
    # Set the blue confidence bars to have minimal height to match example
    plt.bar(bar_positions - bar_width/2, stats_df['Avg Confidence'], 
            width=bar_width, color='skyblue', label='Avg Confidence')
    plt.bar(bar_positions + bar_width/2, stats_df['Directional Accuracy'], 
            width=bar_width, color='orange', label='Directional Accuracy')
    
    # Add sample size labels above bars
    for i, pos in enumerate(bar_positions):
        plt.text(pos, 0.6, f"n={stats_df['Sample Size'].iloc[i]}", 
                ha='center')
    
    # Add gridlines
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', color='#E0E0E0')
    
    # Customize plot
    plt.xlabel(f'{asset_class.upper()} Tickers')
    plt.ylabel('Value')
    plt.title(f'Average Confidence vs Directional Accuracy by {asset_class.upper()} Ticker')
    plt.xticks(bar_positions, stats_df['Ticker'])
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'data/evaluation/{asset_class}/{asset_class}_directional_accuracy_by_ticker.png')
    
    # Print statistics for reference
    print("\nETF Performance Statistics:")
    # Format the Avg Confidence column to show percentages with 2 decimal places
    stats_df_display = stats_df.copy()
    stats_df_display['Avg Confidence'] = stats_df_display['Avg Confidence'].apply(lambda x: f"{x*100:.2f}%")
    print(stats_df_display.to_string(index=False))
    

# ------------------------------------------------------------------
# 4.  PLOT – average across all sectors
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# lead-lag IC diagnostic  (-2 … +2 weeks)
# ------------------------------------------------------------------
lags   = range(-2, 3)                                # −2 = view made 2 wks *after* the realised
ic_lag = {}
for k in lags:
    r_shift = realised.shift(-k)                     # negative → push actuals *forward*
    ic_lag[k] = (pred.rank(axis=1)
                 .corrwith(r_shift.rank(axis=1), axis=1, method='spearman')).mean()

print("Mean Spearman IC by lag:"); print(ic_lag)

# any ETF column mostly NaN?
coverage = pred.notna().mean().sort_values()
print("\nPrediction coverage:")
print(coverage[coverage < 0.9])

BEST_LAG = np.argmax(ic_lag)        # replace with the argmax from ic_lag

real_aligned = realised.shift(-BEST_LAG).loc[pred.index]

avg_pred = pred.mean(axis=1)
avg_real = real_aligned.mean(axis=1)

plt.figure(figsize=(11,4))
plt.plot(avg_pred, label='Predicted')
plt.plot(avg_real, label='Realised', alpha=.7)
plt.axhline(0, ls='--', c='grey', lw=.6)
plt.title(f'Weekly Return (lag = {BEST_LAG})')
plt.legend()

