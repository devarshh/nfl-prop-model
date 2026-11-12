"""Scores backtest_all.parquet (from backtest.py).

MAE: error of the median against simple averages. Calibration: how often the result landed below each
predicted quantile (should match the level). CRPS: squared error of the predicted over/under probability summed
over every x.5 line (lower is better; it rewards being right about the whole distribution, not just the median).
"""
import numpy as np
import pandas as pd

from model import QUANTILES, prob_over, qcol, cdf

LINES = {"passing_yards": np.arange(-50.5, 650, 1.0), "rushing_yards": np.arange(-30.5, 320, 1.0),
         "receiving_yards": np.arange(-20.5, 320, 1.0), "receptions": np.arange(-0.5, 25, 1.0)}

d = pd.read_parquet("backtest_all.parquet")
qc = [qcol(q) for q in QUANTILES]
rows = []
for mk, g in d.groupby("market"):
    y = g["actual"].to_numpy(float)
    Q = g[qc].to_numpy(float)
    r = {"market": mk, "n": len(g), "MAE_model": np.mean(np.abs(y - g[qcol(0.5)]))}
    for c in ["l5_mean", "l5_median", "season_avg", "ewm"]:
        r[f"MAE_{c}"] = np.mean(np.abs(y - g[c].fillna(g["l16"])))
    for q in QUANTILES:
        r[f"below_{qcol(q)}"] = np.mean(y < g[qcol(q)])
    grid = LINES[mk]
    r["CRPS"] = np.mean([np.sum((cdf(np.repeat(Q[i:i + 1], len(grid), 0), grid) - (y[i] <= grid)) ** 2)
                         for i in range(len(g))])
    rows.append(r)
res = pd.DataFrame(rows).set_index("market")
print(res.round(3).T.to_string())

# Naive line test: pretend the book line = last-5 average, rounded to x.5 (real books are much sharper)
print("\nNaive-line test (line = last-5 average):")
for mk, g in d.groupby("market"):
    line = np.floor(g["l5_mean"].fillna(g["l16"])) + 0.5
    p = prob_over(g[qc], line, mk)
    over = g["actual"].to_numpy() > line.to_numpy()
    hit = (p > 0.5) == over
    conf = (p > 0.6) | (p < 0.4)
    brier = np.mean((p - over) ** 2)
    print(f"  {mk:16s} actual over naive line {over.mean():.1%} | model side hit {hit.mean():.1%} (n={len(g)})"
          f" | confident picks hit {hit[conf].mean():.1%} (n={conf.sum()}) | Brier {brier:.4f}")
