"""
LightGBM quantile models for NFL player props.
Markets: passing yards, rushing yards, receiving yards, receptions.

One model per quantile level. Between the predicted quantiles the CDF is a straight line; beyond the outermost
ones it has exponential tails, so P(over) is defined for any line (week_template.html uses the same formula).
"""
import json
import platform
import subprocess

import numpy as np
import pandas as pd
import lightgbm as lgb
from features import feature_list

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
MARKETS = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]

# Whole-number stats are modeled as stat + uniform(-0.5, 0.5) ("jittering"). That makes the quantiles of a count
# well defined, and P(over 4.5) is then exactly 1 - F(4.5) instead of an interpolation between whole numbers.
JITTER = {"receptions"}


def _threads():
    """On Apple silicon LightGBM is ~1.5x faster on the performance cores alone than on every core."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                 capture_output=True, text=True, timeout=2).stdout
            return max(int(out), 1)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return 0  # LightGBM default: all cores


PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=80, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=63, verbose=-1,
              num_threads=_threads())


def qcol(q):
    return f"q{q * 100:g}"  # 0.05 -> "q5", 0.5 -> "q50"


def market_masks(df, market):
    """(train_mask, eval_mask). Eval = the kind of player a sportsbook actually posts a line for."""
    pos, has_hist = df["position"], df["career_games"] > 0
    if market == "passing_yards":
        tr = (pos == "QB") & (df["is_starting_qb"] == 1)
        ev = tr & has_hist & (df["attempts_ewm"] >= 15)
    elif market == "rushing_yards":
        rb = pos.isin(["RB", "FB"])
        qb = (pos == "QB") & (df["is_starting_qb"] == 1)
        tr = (rb & (df["carries_ewm"].fillna(0) >= 2)) | qb | (rb & ~has_hist)
        ev = has_hist & ((rb & (df["carries_ewm"] >= 6)) | (qb & (df["rushing_yards_ewm"] >= 15)))
    else:  # receiving_yards / receptions
        sk = pos.isin(["WR", "TE", "RB"])
        tr = sk & ((df["targets_ewm"].fillna(0) >= 1) | ~has_hist)
        ev = sk & has_hist & (df["targets_ewm"] >= 3.5)
    return tr, ev


def train_target(y, market, seed=0):
    y = np.asarray(y, dtype=float)
    if market in JITTER:
        y = y + np.random.default_rng(seed).uniform(-0.5, 0.5, len(y))
    return y


def load_iters(path="iters.json"):
    """market -> {quantile level: number of trees}. A level missing from the file (e.g. an iters.json from an
    older grid) borrows the count of the nearest level that is there."""
    out = {}
    for mk, v in json.load(open(path)).items():
        have = {float(q): int(n) for q, n in v.items()} if isinstance(v, dict) else {0.5: int(v)}
        out[mk] = {q: have[min(have, key=lambda h: abs(h - q))] for q in QUANTILES}
    return out


def fit_quantiles(X, y, market, iters):
    """One LightGBM quantile model per level (iters: level -> trees), all on the same binned dataset."""
    ds = lgb.Dataset(X, train_target(y, market), free_raw_data=False)
    return {q: lgb.train(dict(PARAMS, objective="quantile", alpha=q), ds, num_boost_round=iters[q])
            for q in QUANTILES}


def predict_quantiles(models, X, market):
    P = np.column_stack([models[q].predict(X) for q in QUANTILES])
    lo = -0.5 if market in JITTER else 0.0
    P = np.sort(np.clip(P, lo, None), axis=1)  # no quantile crossing, no impossible values
    return pd.DataFrame(P, columns=[qcol(q) for q in QUANTILES], index=X.index)


def cdf(Q, x, levels=QUANTILES):
    """P(stat <= x) for each row of predicted quantiles Q (rows sorted ascending)."""
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    x = np.broadcast_to(np.asarray(x, dtype=float), (len(Q),))
    lv = np.asarray(levels, dtype=float)
    out = np.empty(len(Q))
    for i, (q, xi) in enumerate(zip(Q, x)):
        if xi < q[0]:
            beta = lv[0] * max(q[1] - q[0], 1e-6) / (lv[1] - lv[0])  # density continuous at q[0]
            out[i] = lv[0] * np.exp(-(q[0] - xi) / beta)
        elif xi > q[-1]:
            beta = (1 - lv[-1]) * max(q[-1] - q[-2], 1e-6) / (lv[-1] - lv[-2])
            out[i] = 1 - (1 - lv[-1]) * np.exp(-(xi - q[-1]) / beta)
        else:
            k = max(int(np.searchsorted(q, xi, side="left")), 1)
            span = q[k] - q[k - 1]
            out[i] = lv[k - 1] + (lv[k] - lv[k - 1]) * (xi - q[k - 1]) / span if span > 0 else lv[k]
    return out


def prob_over(qdf, line, market=None):
    """P(stat > line). For receptions on a whole-number line (push possible) this is P(over | no push)."""
    Q = np.asarray(qdf, dtype=float)
    line = np.broadcast_to(np.asarray(line, dtype=float), (len(Q),))
    if market in JITTER:
        whole = np.isclose(line, np.round(line))
        over = 1 - cdf(Q, np.where(whole, line + 0.5, line))
        under = cdf(Q, np.where(whole, line - 0.5, line))
        return np.where(whole, over / np.maximum(over + under, 1e-9), over)
    return 1 - cdf(Q, line)


def tune(df, val_season=None):
    """Trees per market and quantile: train on earlier seasons, take the minimum of each quantile's full
    validation curve on val_season (the last complete season by default).

    The tails of the passing model overfit within ~50-100 trees while its median wants ~200-400, so each
    quantile gets its own count.
    """
    if val_season is None:  # last season with playoff games, i.e. the last complete one
        val_season = int(df.loc[df["is_post"] == 1, "season"].max())
    iters = {}
    for mk in MARKETS:
        feats = feature_list(df, mk)
        tr, _ = market_masks(df, mk)
        a = tr & (df["season"] < val_season)
        b = tr & (df["season"] == val_season)
        ds = lgb.Dataset(df.loc[a, feats], train_target(df.loc[a, mk], mk), free_raw_data=False)
        dv = lgb.Dataset(df.loc[b, feats], train_target(df.loc[b, mk], mk, seed=1), reference=ds)
        iters[mk] = {}
        for q in QUANTILES:
            ev = {}
            lgb.train(dict(PARAMS, objective="quantile", alpha=q, metric="quantile"), ds, num_boost_round=1500,
                      valid_sets=[dv], valid_names=["val"], callbacks=[lgb.record_evaluation(ev)])
            iters[mk][str(q)] = int(np.argmin(ev["val"]["quantile"])) + 1
        print(f"{mk}: {iters[mk]} (validated on {val_season})", flush=True)
    return iters


if __name__ == "__main__":
    df = pd.read_parquet("features.parquet")
    json.dump(tune(df), open("iters.json", "w"), indent=1)
