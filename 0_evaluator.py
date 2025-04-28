import pandas as pd, numpy as np, yfinance as yf, matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
# ------------------------------------------------------------------
# 1.  CONFIG & DATA
# ------------------------------------------------------------------
from src.config.settings import PORTFOLIOS  

asset_class = "equity"  # fx, equity, bond, commodity
PRED_FILE   = f"data/{asset_class}_weekly_predictions.csv"  
PRICE_FILE  = f"data/{asset_class}_combined_features_weekly.csv"       # already saved
fx_tickers = [entry["etf"] for entry in PORTFOLIOS['fx'].get("currencies", [])]
fi_tickers = [entry["etf"] for entry in PORTFOLIOS['bond'].get("treasuries", [])]
equity_tickers = [entry["etf"] for entry in PORTFOLIOS["equity"]["sectors"]]
commodity_tickers = [entry["etf"] for entry in PORTFOLIOS["commodity"]["sectors"]]

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

# date range that matches your back-test:
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

weekly_px = px.resample("W-FRI").last().loc[START:END]
realised  = weekly_px.pct_change().shift(-1).loc[pred.index]

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
# 3.  PLOT – average across all sectors
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
plt.legend(); plt.show()
