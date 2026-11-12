"""Reality check: grade every final game of a week's saved projections against the box scores.
usage: python review_week.py <week> [--season 2026] [--bets ~/Downloads/week9_bets.csv]   (from the project folder)
--bets grades the lines you entered on the page (its "Download my bets" button) by verdict: Bet / Lean / Pass."""
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model import cdf, QUANTILES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("week", type=int)
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--bets", help="CSV downloaded from the projections page")
ARGS = ap.parse_args()
WEEK, SEASON = ARGS.week, ARGS.season
STAT = {"pass": "passing_yards", "rush": "rushing_yards", "recyd": "receiving_yards", "rec": "receptions"}
IDX = {q: i for i, q in enumerate(QUANTILES)}
LINES = {"pass": np.arange(0.5, 600), "rush": np.arange(0.5, 250), "recyd": np.arange(0.5, 250),
         "rec": np.arange(0.5, 20)}

games = json.load(open(f"output/week{WEEK}.json"))
final = [g for g in games if g.get("final")]
ps = pd.read_parquet("ps.parquet")
ps = ps[(ps.season == SEASON) & (ps.week == WEEK)]
sc = pd.read_parquet("sc.parquet")
sc = sc[(sc.season == SEASON) & (sc.week == WEEK)]


def pctile(q, y, k):
    Q = np.array([q], float)
    if k == "rec":
        return 50 * (cdf(Q, y - 0.5)[0] + cdf(Q, y + 0.5)[0])
    return 100 * cdf(Q, y)[0]


def crps(q, y, k):
    """Brier score summed over every x.5 line (what the backtests use)."""
    L = LINES[k]
    p_under = cdf(np.array([q], float).repeat(len(L), 0), L) if False else np.array(
        [cdf(np.array([q], float), x)[0] for x in L])
    return float(((1 - p_under) - (y > L)) ** 2 @ np.ones(len(L)))


rows, tds, dnp = [], [], []
for g in final:
    f = g["final"]
    home_margin = f["home"] - f["away"]
    for p in g["players"]:
        a = p.get("a")
        side = "home" if p["t"] == g["home"] else "away"
        exp_margin = g["spread"] if side == "home" else -g["spread"]
        act_margin = home_margin if side == "home" else -home_margin
        team_pts = f[side]
        team_tt = g[f"{side}_tt"]
        if a is None:
            dnp.append((g["id"], p["n"], p["t"], p["p"], p["f"], list(p["m"])))
            continue
        for k, q in p["m"].items():
            y = a[k]
            r = dict(game=g["id"], name=p["n"], team=p["t"], pos=p["p"], flag=p["f"], new=p["new"], k=k, y=y,
                     med=q[IDX[0.5]], q10=q[IDX[0.1]], q90=q[IDX[0.9]], q25=q[IDX[0.25]], q75=q[IDX[0.75]],
                     pct=pctile(q, y, k), crps_blend=crps(q, y, k),
                     script=act_margin - exp_margin, pts_vs_tt=team_pts - team_tt)
            if k in p.get("g", {}):
                r["crps_stats"] = crps(p["g"][k], y, k)
            if k in p.get("s", {}):
                r["crps_sim"] = crps(p["s"][k], y, k)
            rows.append(r)
        if "td" in p:
            tds.append(dict(game=g["id"], name=p["n"], team=p["t"], pos=p["p"], p_any=p["td"]["m"][0],
                            p_stats=p["td"]["g"][0], p_sim=p["td"].get("s", [np.nan])[0], scored=int(a["td"] > 0)))

R = pd.DataFrame(rows)
T = pd.DataFrame(tds)
R.to_csv(f"output/week{WEEK}_graded.csv", index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

print(f"===== {len(final)} final games, {len(R)} props, {len(dnp)} projected players with no stat line")
print("\n--- Games: actual vs market")
for g in final:
    f = g["final"]
    print(f'{g["away"]:>3} {f["away"]:>2} ({g["away_tt"]:>4}) @ {g["home"]:>3} {f["home"]:>2} ({g["home_tt"]:>4}) '
          f'total {f["away"] + f["home"]:>2} vs {g["total"]:<5} home margin {f["home"] - f["away"]:+3} vs {g["spread"]:+}')

print("\n--- Calibration by stat (ideal: mean pct 50, 50% in middle-50, 80% in middle-80, 50% over median)")
cal = R.assign(in50=(R.y >= R.q25) & (R.y <= R.q75), in80=(R.y >= R.q10) & (R.y <= R.q90),
               over=R.y > R.med, err=R.y - R.med).groupby("k").agg(
    n=("y", "size"), mean_pct=("pct", "mean"), in50=("in50", "mean"), in80=("in80", "mean"),
    over_med=("over", "mean"), mean_err=("err", "mean"), median_err=("err", "median"),
    blend=("crps_blend", "mean"), stats=("crps_stats", "mean"), sim=("crps_sim", "mean"))
print(cal.round(3))
print("CRPS relative to stats model (negative = better):")
for k, c in cal.iterrows():
    print(f"  {k:5} blend {100 * (c.blend / c.stats - 1):+.1f}%   sim {100 * (c.sim / c.stats - 1):+.1f}%")

print("\n--- Mean percentile by position x stat")
print(R.pivot_table(index="pos", columns="k", values="pct", aggfunc=["mean", "size"]).round(1))

print("\n--- Game script: correlation of percentile with (actual - expected margin) and (team pts - implied)")
for k in STAT:
    s = R[R.k == k]
    print(f"  {k:5} margin r={s.pct.corr(s.script):+.2f}  team-points r={s.pct.corr(s.pts_vs_tt):+.2f}  n={len(s)}")

print("\n--- Team offense: sum of actual minus sum of medians, by team (yards props)")
team = R[R.k != "rec"].assign(err=R.y - R.med).pivot_table(index="team", columns="k", values="err", aggfunc="sum")
pts = R.drop_duplicates("team").set_index("team").pts_vs_tt
print(team.join(pts).round(0).sort_values("pts_vs_tt"))

print("\n--- New-starter QBs and their receivers")
newteams = set(R[R.new].team)
print("new-starter teams:", newteams)
print(R[R.team.isin(newteams)].groupby("k").pct.agg(["mean", "size"]).round(1))
print(R[R.new][["name", "team", "k", "y", "med", "pct"]])

print("\n--- Flags")
print(R.groupby(R.flag.replace("", "(none)")).pct.agg(["mean", "size"]).round(1))

print("\n--- Biggest misses (outside the 5-95 band)")
cols = ["game", "name", "team", "pos", "flag", "k", "y", "q10", "med", "q90", "pct"]
print(R[(R.pct < 5) | (R.pct > 95)].sort_values("pct")[cols].round(1).to_string(index=False))

print("\n--- Projected players with no stat line (DNP / inactive)")
for d in dnp:
    print("  ", d)

print("\n--- Touchdowns")
from math import log
T["ll"] = -(T.scored * np.log(T.p_any) + (1 - T.scored) * np.log(1 - T.p_any))
print(f"n={len(T)} expected scorers {T.p_any.sum():.1f}, actual {T.scored.sum()}; "
      f"log loss blend {T.ll.mean():.4f}; Brier blend {((T.p_any - T.scored) ** 2).mean():.4f} "
      f"stats {((T.p_stats - T.scored) ** 2).mean():.4f} sim {((T.p_sim - T.scored) ** 2).mean():.4f}")
print(T.groupby("pos").agg(n=("scored", "size"), exp=("p_any", "sum"), act=("scored", "sum")).round(1))
T["bucket"] = pd.cut(T.p_any, [0, .1, .2, .3, .4, .5, 1])
print(T.groupby("bucket", observed=True).agg(n=("scored", "size"), pred=("p_any", "mean"), hit=("scored", "mean")).round(3))
print("Top-3 likeliest scorers per game who scored:", end=" ")
top = T.sort_values("p_any", ascending=False).groupby("game").head(3)
print(f"{top.scored.sum()}/{len(top)} (expected {top.p_any.sum():.1f})")

# all actual TD scorers (rush/rec) this week, and whether we listed them
box = ps[ps.game_id.isin([g["id"] for g in final])]
scorers = box[(box.rushing_tds + box.receiving_tds) > 0]
listed = {(r["name"], r["team"]): r["p_any"] for _, r in T.iterrows()}
tdg = set(T.game); print("Scorers we did not list (games with TD projections):", [(r.player_display_name, r.team, r.position) for r in scorers[scorers.game_id.isin(tdg)].itertuples()
                                    if (r.player_display_name, r.team) not in listed])

print("\n--- Parlays (suggested legs graded against box scores)")
act = {}
for g in final:
    for p in g["players"]:
        if p.get("a"):
            act[p.get("id")] = p["a"]
summ = {}
for g in final:
    for key, par in (g.get("parlays") or {}).items():
        hit = True
        for lg in par["legs"]:
            a = act.get(lg["id"])
            if a is None:
                hit = False
                break
            if lg["stat"] in ("td", "td2", "any", "two") or "td" in lg["stat"]:
                need = 2 if "2" in lg["stat"] else 1
                hit &= a.get("td", 0) >= need
            else:
                hit &= a.get(lg["stat"], -1) >= lg["x"]
        summ.setdefault(key, []).append((par["p"], hit))
for key in sorted(summ, key=lambda s: (s.count("same"), s)):
    v = summ[key]
    print(f"  {key:14} n={len(v):2}  predicted {np.mean([x[0] for x in v]):.2f}  hit {np.mean([x[1] for x in v]):.2f}")

print("\n--- Unprojected players with notable lines")
proj_ids = {p.get("id") for g in final for p in g["players"]}
big = box[~box.player_id.isin(proj_ids) & ((box.receiving_yards >= 50) | (box.rushing_yards >= 40) |
                                            (box.passing_yards >= 100) | (box.receptions >= 5))]
print(big[["game_id", "player_display_name", "team", "position", "attempts", "passing_yards", "carries",
           "rushing_yards", "targets", "receptions", "receiving_yards"]].to_string(index=False))

print("\n--- Projected players with low snap share (<35% offense: early exit / limited)")
low = sc[sc.offense_pct < 0.35]
names = {(p["n"], p["t"]) for g in final for p in g["players"] if p.get("a")}
print(low[[ (r.player, r.team) in names for r in low.itertuples()]][["game_id", "player", "team", "position", "offense_snaps", "offense_pct"]].to_string(index=False))


def profit(odds, result):
    """Profit on a $1 stake: American (+150/-120) or decimal (2.50) odds; push = 0, loss = -1."""
    if result != "win":
        return 0.0 if result == "push" else -1.0
    o = float(str(odds).replace("\u2212", "-"))
    if abs(o) >= 100:
        return o / 100 if o > 0 else 100 / -o
    return o - 1


if ARGS.bets:
    import os
    B = pd.read_csv(os.path.expanduser(ARGS.bets), dtype={"over_odds": str, "under_odds": str, "odds": str})
    box = ps.set_index("player_id")
    by_name = ps.set_index(["player_display_name", "team"])
    out = []
    for r in B.itertuples():
        try:
            a = box.loc[r.player_id] if isinstance(r.player_id, str) and r.player_id in box.index else by_name.loc[(r.player, r.team)]
        except KeyError:
            out.append(dict(r._asdict(), actual=np.nan, result="no stat line (void)", profit=0.0))
            continue
        if isinstance(a, pd.DataFrame):
            a = a.iloc[0]
        if r.market == "td":
            y = a.rushing_tds + a.receiving_tds
            res = "win" if y > 0 else "loss"
            odds = r.odds
        else:
            y = a[STAT[r.market]]
            over = r.side == "over"
            res = "push" if y == r.line else "win" if (y > r.line) == over else "loss"
            odds = r.over_odds if over else r.under_odds
        out.append(dict(r._asdict(), actual=y, result=res, profit=profit(odds, res)))
    G = pd.DataFrame(out)
    print(f"\n===== Your entered bets ({len(G)})")
    print(G[["player", "market", "line", "side", "verdict", "p_model", "break_even", "actual", "result", "profit"]]
          .round(3).to_string(index=False))
    graded = G[G.result.isin(["win", "loss", "push"])]
    summ = graded.groupby("verdict").agg(n=("result", "size"), wins=("result", lambda x: (x == "win").sum()),
                                         pushes=("result", lambda x: (x == "push").sum()),
                                         exp_win=("p_model", "mean"), units=("profit", "sum"))
    summ["roi"] = summ.units / summ.n
    print("\nBy verdict (1 unit each; units = profit, roi = per unit staked):")
    print(summ.round(3))
    G.to_csv(f"output/week{WEEK}_bets_graded.csv", index=False)
    print(f"Saved output/week{WEEK}_bets_graded.csv. One week is far too few bets to judge the model: "
          "keep these files and pool them over the season.")
