"""
Anytime-touchdown projections: the chance each player scores a rushing or receiving touchdown.

Two methods, as for the yardage props, mixed with a weight chosen in a walk-forward backtest:
  - stats model: a LightGBM Poisson model of the player's touchdowns in the game (rushing + receiving). Inputs:
    his touchdowns and expected touchdowns over recent games (nflverse rates every carry and target by its odds
    of scoring from where on the field it came, so this is his red-zone and goal-line role), his share of what
    the active players on his team usually get, his team's implied points, the touchdowns his team scores and
    the opponent allows, and the yardage models' inputs. Touchdowns in a game are close to Poisson, so
    P(anytime) = 1 - exp(-rate) and P(2+) = 1 - exp(-rate) * (1 + rate).
  - game simulation: the share of the 1,000 simulated games in which he scored (simulate.py). It also gives the
    chance he scores the game's first touchdown, which the stats model can't.

    python touchdowns.py tune                   # number of trees, validated on the last complete season
    python touchdowns.py backtest 2025 2026     # stats model week by week -> td_backtest.parquet (a few minutes)
    python touchdowns.py blend                  # + sim_td_backtest.parquet (backtest_sim.py): scores both
                                                #   methods and writes the weight to blend.json["td"]

Return touchdowns and 2-point conversions are not counted. (Sportsbooks count return TDs for anytime bets;
they're rare for anyone but kick returners.)
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import PROJECTIONS, feature_list
from model import PARAMS

TD_ROLL = ["tds", "rushing_tds", "receiving_tds", "rec_touchdown_exp", "rush_touchdown_exp", "td_exp"]
TD_EXTRA = ["td_exp_share", "tds_share", "x_td_vegas", "x_td_team", "team_tds_tm", "team_tdexp_tm",
            "opp_tds_allowed", "opp_tdexp_allowed"]
TD_PARAMS = dict(PARAMS, objective="poisson", num_leaves=15, min_child_samples=200)
TD_TREES = 645  # python touchdowns.py tune 2024 (trained on 2016-23; the curve is flat from ~300 to 1000)


def p_any(lam):
    return 1.0 - np.exp(-np.asarray(lam, dtype=float))


def p_two(lam):
    lam = np.asarray(lam, dtype=float)
    return 1.0 - np.exp(-lam) * (1.0 + lam)


def add_td_features(df, path="."):
    """Touchdown features, built only from earlier games. Returns df with the new columns (same index/order)."""
    ffo = pd.read_parquet(f"{path}/ffopp.parquet",
                          columns=["game_id", "player_id", "rec_touchdown_exp", "rush_touchdown_exp"])
    x = df[["game_id", "player_id", "team", "opponent_team", "seq", "rushing_tds", "receiving_tds",
            "implied_total"]].merge(ffo.drop_duplicates(["game_id", "player_id"]), on=["game_id", "player_id"],
                                    how="left")
    x.index = df.index  # left merge on unique keys keeps df's rows in order
    x[["rec_touchdown_exp", "rush_touchdown_exp"]] = x[["rec_touchdown_exp", "rush_touchdown_exp"]].fillna(0.0)
    x["tds"] = x["rushing_tds"] + x["receiving_tds"]
    x["td_exp"] = x["rec_touchdown_exp"] + x["rush_touchdown_exp"]
    # (a future game's own stats are unknown, i.e. 0; every feature below uses earlier games only)

    x = x.sort_values(["player_id", "seq"])
    pid = x["player_id"]
    sh = x.groupby("player_id")[TD_ROLL].shift(1)
    new = {}
    for c in TD_ROLL:
        gx = sh[c].groupby(pid)
        new[f"{c}_ewm"] = gx.ewm(halflife=4, min_periods=1).mean().reset_index(level=0, drop=True)
        new[f"{c}_l16"] = gx.rolling(16, min_periods=1).mean().reset_index(level=0, drop=True)
        new[f"{c}_car"] = gx.expanding().mean().reset_index(level=0, drop=True)
    x = pd.concat([x, pd.DataFrame(new, index=x.index)], axis=1)

    # share of what this game's active players usually get (teammates out -> more for the rest)
    for c, name in (("td_exp_ewm", "td_exp_share"), ("tds_l16", "tds_share")):
        tot = x.groupby(["game_id", "team"])[c].transform("sum")
        x[name] = x[c] / tot.replace(0, np.nan)

    # team scoring and what the opponent allows, rolled over earlier games
    def rolled(by, cols, names):
        t = x.groupby(["game_id", by, "seq"], as_index=False)[cols].sum().sort_values([by, "seq"])
        s = t.groupby(by)[cols].shift(1)
        for c, nm in zip(cols, names):
            t[nm] = s[c].groupby(t[by]).ewm(halflife=6, min_periods=1).mean().reset_index(level=0, drop=True)
        return t[["game_id", by] + names]
    tm = rolled("team", ["tds", "td_exp"], ["team_tds_tm", "team_tdexp_tm"])
    op = rolled("opponent_team", ["tds", "td_exp"], ["opp_tds_allowed", "opp_tdexp_allowed"])
    x.index.name = "_row"
    x = x.reset_index().merge(tm, on=["game_id", "team"], how="left") \
         .merge(op, on=["game_id", "opponent_team"], how="left").set_index("_row")
    x["x_td_vegas"] = x["td_exp_share"] * x["implied_total"]
    x["x_td_team"] = x["td_exp_share"] * x["team_tdexp_tm"]

    keep = ["tds", "td_exp"] + list(new) + TD_EXTRA
    out = df.drop(columns=[c for c in keep if c in df.columns])
    return pd.concat([out, x.loc[df.index, keep]], axis=1)


def td_features(df):
    return feature_list(df) + PROJECTIONS + [f"{c}_{k}" for c in TD_ROLL for k in ("ewm", "l16", "car")] + TD_EXTRA


def td_masks(df):
    """(train, eval): train on every skill player who played (QBs: the starter); evaluate on the players a
    sportsbook posts anytime-TD odds for."""
    pos = df["position"]
    tr = pos.isin(["RB", "FB", "WR", "TE"]) | ((pos == "QB") & (df["is_starting_qb"] == 1))
    ev = tr & (df["career_games"] > 0) & ((df["targets_ewm"] >= 1.5) | (df["carries_ewm"] >= 2)
                                          | (df["is_starting_qb"] == 1))
    return tr, ev


def fit_td(X, y, trees=TD_TREES):
    return lgb.train(TD_PARAMS, lgb.Dataset(X, np.asarray(y, dtype=float)), num_boost_round=trees)


def tune(df, val_season=None):
    if val_season is None:
        val_season = int(df.loc[df["is_post"] == 1, "season"].max())
    feats, (tr, _) = td_features(df), td_masks(df)
    a, b = tr & (df["season"] < val_season), tr & (df["season"] == val_season)
    ds = lgb.Dataset(df.loc[a, feats], df.loc[a, "tds"].astype(float), free_raw_data=False)
    dv = lgb.Dataset(df.loc[b, feats], df.loc[b, "tds"].astype(float), reference=ds)
    ev = {}
    m = lgb.train(dict(TD_PARAMS, metric="poisson"), ds, num_boost_round=2000, valid_sets=[dv],
                  valid_names=["val"], callbacks=[lgb.record_evaluation(ev)])
    curve = np.asarray(ev["val"]["poisson"])
    best = int(np.argmin(curve)) + 1
    y = (df.loc[b, "tds"] >= 1).to_numpy()
    for n in sorted({100, 200, 300, best, 600, 1000}):
        p = np.clip(p_any(m.predict(df.loc[b, feats], num_iteration=n)), 1e-4, 1 - 1e-4)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        print(f"  {n:5d} trees: poisson {curve[n - 1]:.5f}, anytime-TD log loss {ll:.5f}")
    print(f"best: {best} trees (validated on {val_season})")
    return best


def backtest(df, seasons, trees=TD_TREES):
    feats, (tr, ev) = td_features(df), td_masks(df)
    out = []
    for s in seasons:
        for w in sorted(df.loc[df["season"] == s, "week"].unique()):
            cut = s * 100 + w
            a, t = tr & (df["seq"] < cut), ev & (df["seq"] == cut)
            if not t.any():
                continue
            m = fit_td(df.loc[a, feats], df.loc[a, "tds"], trees)
            r = df.loc[t, ["game_id", "season", "week", "player_id", "player_display_name", "position", "team",
                           "tds", "rushing_tds", "receiving_tds", "tds_ewm", "td_exp_ewm", "tds_l16"]].copy()
            r["lam"] = m.predict(df.loc[t, feats])
            out.append(r)
            print(f"{s} week {w}: trained on {a.sum()}, predicted {t.sum()}", flush=True)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------
# Scoring the two methods and choosing the blend weight
# ---------------------------------------------------------------------------------------------------------------
def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def first_scorers(pbp):
    """game_id -> player_id of the game's first touchdown (None when a defense or return unit scored it)."""
    p = pbp[pbp["touchdown"] == 1].sort_values(["game_id", "game_seconds_remaining", "play_id"],
                                               ascending=[True, False, True]).groupby("game_id").head(1)
    who = np.where(p["pass_touchdown"] == 1, p["receiver_player_id"],
                   np.where(p["rush_touchdown"] == 1, p["rusher_player_id"], None))
    return dict(zip(p["game_id"], who))


def blend_main():
    st = pd.read_parquet("td_backtest.parquet")
    sm = pd.read_parquet("sim_td_backtest.parquet")
    m = st.merge(sm, on=["game_id", "player_id"], how="inner")
    seasons = sorted(m["season"].unique())
    fit, last = m["season"].isin(seasons[:-1] or seasons).to_numpy(), m["season"].eq(seasons[-1]).to_numpy()
    y, y2 = (m["tds"] >= 1).to_numpy(float), (m["tds"] >= 2).to_numpy(float)
    pg, ps = p_any(m["lam"]), np.clip(m["p_any"].to_numpy(float), 0.001, 0.999)
    print(f"{len(m)} player-games matched ({m.game_id.nunique()} games, seasons {seasons}); "
          f"scored a TD: {y.mean():.1%}, 2+: {y2.mean():.1%}\n")
    base = {"position average": m.groupby("position")["tds"].transform("mean").pipe(p_any),
            "his recent TD rate": p_any(m["tds_ewm"].fillna(0)),
            "his expected TDs (red-zone role)": p_any(m["td_exp_ewm"].fillna(0))}
    print(f"{'anytime TD':34s} {'log loss':>9s} {'Brier':>8s}   (last season: {seasons[-1]})")
    for nm, p in list(base.items()) + [("stats model", pg), ("game simulation", ps)]:
        print(f"  {nm:32s} {logloss(p, y):9.4f} {brier(p, y):8.4f}   {logloss(p[last], y[last]):.4f}")
    W = np.round(np.arange(0, 1.01, 0.1), 2)
    score = {w: (1 - w) * pg + w * ps for w in W}
    w_fit = min(W, key=lambda w: logloss(score[w][fit], y[fit]))
    w_all = min(W, key=lambda w: logloss(score[w], y))
    gain = (logloss(pg[last], y[last]) - logloss(score[w_fit][last], y[last])) / logloss(pg[last], y[last]) * 100
    # paired check on the last season, clustered by game
    d = pd.Series([logloss(score[w_fit][i:i + 1], y[i:i + 1]) - logloss(pg[i:i + 1], y[i:i + 1])
                   for i in np.flatnonzero(last)]).groupby(m["game_id"].to_numpy()[last]).sum()
    boot = [d.sample(frac=1, replace=True, random_state=i).sum() / last.sum() for i in range(500)]
    se = float(np.std(boot) / logloss(pg[last], y[last]) * 100)
    print(f"\nblend: weight on the simulation chosen on {seasons[:-1] or seasons} = {w_fit}; on {seasons[-1]} "
          f"it scores {logloss(score[w_fit][last], y[last]):.4f} vs stats model {logloss(pg[last], y[last]):.4f} "
          f"({gain:+.2f}% ± {se:.2f}%)")
    print(f"production weight (all seasons): {w_all}")
    pb = score[w_all]
    print("\ncalibration of the blend (all seasons):")
    bins = pd.cut(pb, [0, .05, .1, .15, .2, .3, .4, .5, .6, 1])
    cal = pd.DataFrame({"pred": pb, "y": y, "g": pg, "s": ps}).groupby(bins, observed=True).agg(
        n=("y", "size"), predicted=("pred", "mean"), actual=("y", "mean"), stats=("g", "mean"), sim=("s", "mean"))
    print(cal.round(3).to_string())
    by_pos = pd.DataFrame({"pos": m["position"], "y": y, "blend": pb, "stats": pg, "sim": ps}).groupby("pos").mean()
    print("\nby position (average predicted vs actual):\n" + by_pos.round(3).to_string())
    two_b = (1 - w_all) * p_two(m["lam"]) + w_all * m["p_two"].to_numpy(float)
    print(f"\n2+ TDs: stats {logloss(p_two(m['lam']), y2):.4f}, sim {logloss(np.clip(m['p_two'], 1e-3, 1), y2):.4f}, "
          f"blend {logloss(two_b, y2):.4f}; predicted {two_b.mean():.3%} vs actual {y2.mean():.3%}")

    # first touchdown scorer (simulation only)
    rep = {"n": int(len(m)), "weight_fit": float(w_fit), "test_gain_pct": float(gain), "test_gain_se_pct": se,
           "logloss_stats": logloss(pg, y), "logloss_sim": logloss(ps, y), "logloss_blend": logloss(pb, y)}
    if os.path.exists("pbp.parquet"):
        fs = first_scorers(pd.read_parquet("pbp.parquet", columns=[
            "game_id", "touchdown", "pass_touchdown", "rush_touchdown", "receiver_player_id", "rusher_player_id",
            "game_seconds_remaining", "play_id"]))
        yf = np.array([fs.get(g) == p for g, p in zip(m["game_id"], m["player_id"])], dtype=float)
        pf = np.clip(m["p_first"].to_numpy(float), 1e-4, 1)
        naive = pb / pd.Series(pb).groupby(m["game_id"].to_numpy()).transform("sum").to_numpy() * 0.93
        print(f"first TD scorer: sim log loss {logloss(pf, yf):.4f} vs anytime-share baseline "
              f"{logloss(naive, yf):.4f}; predicted {pf.mean():.4f} vs actual {yf.mean():.4f}")
        fb = pd.cut(pf, [0, .02, .04, .06, .08, .1, .15, 1])
        print(pd.DataFrame({"p": pf, "y": yf}).groupby(fb, observed=True).agg(
            n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3).to_string())
        rep["first_logloss_sim"], rep["first_logloss_naive"] = logloss(pf, yf), logloss(naive, yf)

    cfg = json.load(open("blend.json")) if os.path.exists("blend.json") else {}
    cfg["td"] = {"sim_weight": float(w_all), "chosen_on": [int(s) for s in seasons], "report": rep}
    json.dump(cfg, open("blend.json", "w"), indent=1)
    print(f"\nwrote blend.json['td']: simulation weight {w_all}")


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "blend":
        blend_main()
        raise SystemExit
    if cmd not in ("tune", "backtest"):
        raise SystemExit(__doc__)
    data = add_td_features(pd.read_parquet("features.parquet"))
    if cmd == "tune":
        tune(data, int(args[0]) if args else None)
    else:
        seasons = [int(a) for a in args] or [int(data["season"].max())]
        res = backtest(data, seasons)
        res.to_parquet("td_backtest.parquet")
        print(f"saved td_backtest.parquet: {len(res)} player-games")
