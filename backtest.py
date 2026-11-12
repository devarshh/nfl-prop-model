"""Walk-forward backtest: for each week, train on everything before it, predict that week.

    python backtest.py                  # every played week of the latest season in features.parquet
    python backtest.py 1 2 3 4          # specific weeks
    python backtest.py --season 2025    # another season (note: iters.json was tuned on the last complete
                                        # season, so backtesting that season flatters the model slightly)
"""
import sys

import pandas as pd

from features import feature_list
from model import MARKETS, QUANTILES, market_masks, fit_quantiles, predict_quantiles, load_iters

args = sys.argv[1:]
df = pd.read_parquet("features.parquet")
season = int(df["season"].max())
if "--season" in args:
    i = args.index("--season")
    season = int(args[i + 1])
    args = args[:i] + args[i + 2:]
weeks = [int(w) for w in args] or sorted(int(w) for w in df.loc[df.season == season, "week"].unique())
iters = load_iters()

out = []
for w in weeks:
    cut = season * 100 + w
    for mk in MARKETS:
        feats = feature_list(df, mk)
        tr, ev = market_masks(df, mk)
        a = tr & (df["seq"] < cut)
        t = ev & (df["seq"] == cut)
        if not t.any():
            continue
        models = fit_quantiles(df.loc[a, feats], df.loc[a, mk], mk, iters[mk])
        q = predict_quantiles(models, df.loc[t, feats], mk)
        keep = ["game_id", "season", "week", "player_id", "player_display_name", "position", "team",
                "opponent_team", mk, f"{mk}_l5", f"{mk}_l5med", f"{mk}_std", f"{mk}_ewm", f"{mk}_l16"]
        r = df.loc[t, keep].rename(columns={mk: "actual", f"{mk}_l5": "l5_mean", f"{mk}_l5med": "l5_median",
                                            f"{mk}_std": "season_avg", f"{mk}_ewm": "ewm", f"{mk}_l16": "l16"})
        r["market"] = mk
        out.append(pd.concat([r, q], axis=1))
        if w == weeks[-1]:
            imp = pd.Series(models[0.5].feature_importance("gain"), index=feats).sort_values(ascending=False)
            imp.to_csv(f"importance_{mk}.csv")
        print(f"{season} week {w} {mk}: trained on {a.sum()}, predicted {t.sum()}", flush=True)
    res = pd.concat(out)
    res.to_parquet("backtest_all.parquet")      # for evaluate.py
    res.to_parquet(f"backtest_{season}.parquet")  # for blend.py (one file per season)
print("quantile levels:", QUANTILES)
