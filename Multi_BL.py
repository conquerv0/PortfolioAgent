#!/usr/bin/env python
"""multi_asset_bl_test.py
Aggregate weekly predictions from FX, Fixed‑Income, Equity, and Commodity agents,
apply a single Black–Litterman update across the entire universe, and back‑test a
multi‑asset portfolio against an equal‑weighted benchmark.

Usage:
    python multi_asset_bl_test.py
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, eigvalsh
from sklearn.covariance import LedoitWolf
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple

from src.agent.DataCollector import DataCollector
from src.config.settings import PORTFOLIOS

# ---------------------------------------------------------------------------
# Hyper‑parameters
LOOKBACK_WEEKS        = 4
ROBUST_LOOKBACK_WEEKS = 260   # ≈5y of weekly data
TAU                   = 0.2
UNCERTAINTY_SCALE     = 0.2
RISK_AVERSION         = 1.0
RISK_FREE_RATE        = 0.0
EPSILON               = 1e-4
LAMBDA_BLEND          = 0.7   # weight on short‑window covariance vs long window

# ---------------------------------------------------------------------------
# Utility: pick asset lists directly from settings
ASSET_LISTS: Dict[str, List[str]] = {
    "fx":       [e['etf'] for e in PORTFOLIOS['fx'].get('currencies', [])],
    "fi":       [e['etf'] for e in PORTFOLIOS['bond'].get('treasuries', [])],
    "equity":   [e['etf'] for e in PORTFOLIOS['equity']['sectors']],
    "commodity":[e['etf'] for e in PORTFOLIOS['commodity']['sectors']],
}

# ---------------------------------------------------------------------------

def load_data(asset_class: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw prediction and price/return data for a given asset class."""
    pred_file   = f"data/predictions/{asset_class}_weekly_predictions.csv"
    actual_file = f"data/features/{asset_class}_combined_features_weekly.csv"

    preds = pd.read_csv(pred_file, parse_dates=['date'])
    actual = pd.read_csv(actual_file)

    # infer the column containing the date in actual
    if 'Unnamed: 0' in actual.columns:
        actual['date'] = pd.to_datetime(actual['Unnamed: 0'])
        actual.drop(columns=['Unnamed: 0'], inplace=True)
    else:
        first = actual.columns[0]
        actual['date'] = pd.to_datetime(actual[first])
        actual.drop(columns=[first], inplace=True)

    return preds, actual

# ---------------------------------------------------------------------------

def build_predictions(preds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return predicted‑return and confidence pivot tables."""
    pred_tbl = preds.pivot(index='date', columns='etf', values='predicted_return')
    conf_tbl = preds.pivot(index='date', columns='etf', values='confidence').fillna(0.5)
    return pred_tbl, conf_tbl

def prices_to_returns(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    prices = df.set_index('date')[tickers]
    return prices.pct_change().dropna()

def load_asset_class(asset_class: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load, process predictions + returns for an asset class."""
    preds, actual = load_data(asset_class)
    asset_list = ASSET_LISTS[asset_class]

    pred_tbl, conf_tbl = build_predictions(preds)

    if asset_class == 'fi':
        # actual already contains weekly returns
        returns = actual.set_index('date')[asset_list]
    else:
        returns = prices_to_returns(actual, asset_list)

    return pred_tbl, conf_tbl, returns, asset_list

# ---------------------------------------------------------------------------

def robust_covariance_estimation(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp):
    collector = DataCollector(full_start_date=start.strftime('%Y-%m-%d'),
                               target_start_date=start.strftime('%Y-%m-%d'),
                               end_date=end.strftime('%Y-%m-%d'))
    price = collector.get_etf_adj_close(tickers, start, end)
    weekly_ret = price.resample('W-FRI').last().pct_change().dropna()
    lw = LedoitWolf().fit(weekly_ret.values)
    Sigma = lw.covariance_
    pi    = weekly_ret.mean().values.reshape(-1,1)
    return Sigma, pi

def mean_variance_weights(Sigma: np.ndarray, mu: np.ndarray):
    invS = inv(Sigma)
    raw  = invS @ mu
    norm = np.sum(np.abs(raw))
    return raw / norm if norm > EPSILON else np.ones_like(raw)/len(raw)

def black_litterman(mu_prior, Sigma_prior, q, confidences):
    omega = np.diag([UNCERTAINTY_SCALE*(1-c)+EPSILON for c in confidences])
    inv_term = inv(TAU*Sigma_prior) + inv(omega)
    return inv(inv_term) @ (inv(TAU*Sigma_prior) @ mu_prior + inv(omega) @ q)

# ---------------------------------------------------------------------------

def rolling_bl_backtest_multi(pred_tbl: pd.DataFrame, conf_tbl: pd.DataFrame,
                              returns: pd.DataFrame, tickers: List[str]):
    """Run weekly rolling BL back‑test on combined universe."""
    import numpy as np  # local import for typing
    results, w_history = [], []

    pred_tbl = pred_tbl[tickers]
    conf_tbl = conf_tbl[tickers]

    for current_date in pred_tbl.index:
        hist = returns.loc[:current_date - pd.Timedelta(days=1)]
        if len(hist) < LOOKBACK_WEEKS:
            continue
        lookback = hist.tail(LOOKBACK_WEEKS).dropna(how='any', axis=1)
        assets = lookback.columns.tolist()

        if len(assets) < 2:
            continue

        pi_short  = lookback.mean().values.reshape(-1,1)
        Sigma_short = lookback.cov().values

        robust_start = current_date - pd.Timedelta(weeks=ROBUST_LOOKBACK_WEEKS)
        Sigma_long, pi_long = robust_covariance_estimation(assets, robust_start, current_date)
        Sigma = LAMBDA_BLEND*Sigma_short + (1-LAMBDA_BLEND)*Sigma_long
        mu_base = pi_long

        # regularise Sigma
        Sigma = np.nan_to_num(Sigma)
        Sigma = 0.5*(Sigma + Sigma.T)
        eig_min = eigvalsh(Sigma).min()
        if eig_min < 0:
            Sigma -= 1.1*eig_min*np.eye(len(Sigma))

        q = pred_tbl.loc[current_date, assets].values.reshape(-1,1)
        conf = conf_tbl.loc[current_date, assets].values

        mu_bl = black_litterman(mu_base, Sigma, q, conf)
        w = mean_variance_weights(Sigma, mu_bl)
        w_history.append(w.flatten())

        # forward one‑week realised returns
        try:
            next_date = returns.index[returns.index.get_loc(current_date) + 1]
        except (KeyError, IndexError):
            continue
        realised = returns.loc[next_date, assets].values

        n = len(assets)
        w_eq = np.ones((n,1))/n
        bl_ret = float(w.T @ realised.reshape(-1,1))
        eq_ret = float(w_eq.T @ realised.reshape(-1,1))

        results.append({'date': current_date, 'bl_portfolio_return': bl_ret,
                        'equal_weighted_return': eq_ret})

    results_df = pd.DataFrame(results).set_index('date')
    return results_df, w_history

# ---------------------------------------------------------------------------

def plot_cumulative(results_df: pd.DataFrame):
    cum = (1+results_df).cumprod() - 1
    plt.figure(figsize=(10,5))
    cum['bl_portfolio_return'].plot(marker='o', label='BL')
    cum['equal_weighted_return'].plot(lw=1.5, linestyle='--', label='Equal‑Weight')
    plt.ylabel('Cumulative Return')
    plt.title('Multi‑Asset Portfolio Back‑test')
    plt.grid(True)
    plt.legend()
    os.makedirs('data/evaluation', exist_ok=True)
    plt.tight_layout()
    plt.savefig('data/evaluation/multi_asset_cumulative_returns.png')
    plt.show()

# ---------------------------------------------------------------------------

def calc_metrics(ret: pd.Series, w_history):
    total = (1+ret).prod() - 1
    n = len(ret)
    ann_ret = (1+total)**(52/n) - 1
    ann_std = ret.std()*np.sqrt(52)
    sharpe = (ann_ret - RISK_FREE_RATE)/ann_std if ann_std>0 else np.nan
    cum = (1+ret).cumprod()
    dd = (cum - cum.cummax())/cum.cummax()
    mdd = dd.min()
    turnovers = [np.sum(np.abs(w_history[i]-w_history[i-1])) for i in range(1,len(w_history))]
    avg_turn = np.mean(turnovers) if turnovers else np.nan
    return {
        'Annual Return': ann_ret,
        'Annual StdDev': ann_std,
        'Sharpe': sharpe,
        'Max Drawdown': mdd,
        'Avg Turnover': avg_turn
    }

# ---------------------------------------------------------------------------

def main():
    # 1. Load every asset class
    pred_tbls, conf_tbls, returns_tbls = [], [], []
    for cls in ['fx','fi','equity','commodity']:
        pred, conf, rets, _ = load_asset_class(cls)
        pred_tbls.append(pred)
        conf_tbls.append(conf)
        returns_tbls.append(rets)

    combined_pred = pd.concat(pred_tbls, axis=1).sort_index()
    combined_conf = pd.concat(conf_tbls, axis=1).reindex_like(combined_pred).fillna(0.5)
    returns = returns_tbls[0]
    for r in returns_tbls[1:]:
        returns = returns.join(r, how='outer')
    returns = returns.sort_index()

    tickers = combined_pred.columns.tolist()

    results, w_hist = rolling_bl_backtest_multi(combined_pred, combined_conf, returns, tickers)

    if results.empty:
        print('No back‑test results — check data availability.')
        return

    os.makedirs('data/evaluation', exist_ok=True)
    results.to_csv('data/evaluation/multi_asset_portfolio_returns.csv')
    plot_cumulative(results)

    metrics = calc_metrics(results['bl_portfolio_return'], w_hist)
    print('Performance Metrics')
    for k,v in metrics.items():
        print(f'  {k}: {v:.4f}')

if __name__ == '__main__':
    main()
