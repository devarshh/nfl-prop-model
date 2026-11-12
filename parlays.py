"""
Most likely parlays: the combination of standard sportsbook "X+" milestone bets (50+ rushing yards, 5+ receptions,
250+ passing yards...) and touchdown bets that hits most often.

Each leg's chance is the blended projection the page shows. Whether legs hit together comes from the game
simulation: in each of the 1,000 simulated games every player's stat line is known, so legs that tend to happen
together (a QB's yards and his top receiver's, a back's yards and his touchdown) are counted as such, and legs
that pull against each other (two backs splitting carries) too. Technically the legs' chances are the marginals
and the simulation supplies the dependence (its ranks, an empirical copula). Each game is simulated on its own, so
legs from different games are independent.

Without a limit the most likely parlay is a stack of near-certain legs that pays nothing, so every leg is capped
(at most 70%, 80% or 90% likely on its own) and the search finds the legs that hit together most often under that
cap: 2-5 legs within a game, or one leg per game across games.

    python parlays.py backtest 2025 2026     # how often suggested parlays hit in past games vs. predicted
                                             # (needs backtest_2025/2026.parquet from backtest.py, td_backtest)
"""
import sys

import numpy as np

from model import cdf

MILESTONES = {"pass": [150, 175, 200, 225, 250, 275, 300, 325, 350],
              "rush": [25, 40, 50, 60, 70, 80, 90, 100, 125],
              "recyd": [25, 40, 50, 60, 70, 80, 90, 100, 125],
              "rec": [2, 3, 4, 5, 6, 7, 8, 9, 10],
              "td": [1, 2]}
UNIT = {"pass": "pass yds", "rush": "rush yds", "recyd": "rec yds", "rec": "receptions"}
SIM_COLS = {"pass": [0], "rush": [3], "recyd": [5], "rec": [6], "td": [8, 9]}  # simulate.py's stat columns
LEVELS = 250             # simulated games are ranked into this many equal groups per player and stat
CAPS = (0.7, 0.8, 0.9)   # the most likely any single leg may be
LEG_COUNTS = (1, 2, 3, 4, 5)
MIN_LEG = 0.05


def sim_levels(sim, G, seed=7):
    """{(player_id, stat): rank group (0..LEVELS-1) of the player's stat in each simulated game}. Ties (equal
    yards, zero touchdowns) are broken at random, so a leg's simulated games can match any blended chance."""
    rng = np.random.default_rng(seed)
    n = sim.n
    out = {}
    for j, pid in enumerate(G["pid"]):
        for stat, cols in SIM_COLS.items():
            x = sim.stats[:, j, cols].sum(-1).astype(float) + rng.random(n)
            rank = np.argsort(np.argsort(x))
            out[(pid, stat)] = (rank * LEVELS // n).astype(np.uint8)
    return out


def leg_label(stat, x):
    if stat == "td":
        return "anytime TD" if x == 1 else f"{x}+ TDs"
    return f"{x}+ {UNIT[stat]}"


def leg_menu(p):
    """Every milestone leg for one player (page/JSON player dict): (stat, threshold, blended chance)."""
    legs = []
    for stat, q in p["m"].items():
        for x in MILESTONES[stat]:
            legs.append((stat, x, float(1.0 - cdf(np.asarray([q]), x - 0.5)[0])))
    if "td" in p:
        legs += [("td", 1, p["td"]["m"][0]), ("td", 2, p["td"]["m"][1])]
    return legs


def _bits(levels, prob):
    """Simulated games in which a leg with this chance hits: the top `prob` share of the player's ranks."""
    t = int(round(LEVELS * (1.0 - prob)))
    mask = levels >= t
    return int.from_bytes(np.packbits(mask).tobytes(), "big"), int(mask.sum())


def candidates(players, levels, cap):
    """One leg per player and stat: the likeliest milestone not above the cap (any other threshold of the same
    stat hits in a subset of its simulated games, so it can never make a parlay more likely)."""
    out = []
    for p in players:
        if p.get("f") == "Questionable":
            continue  # might not play: a voided leg changes the parlay
        rc = p.get("rc")
        if rc and rc[1] < rc[0] and (rc + ["Snaps"])[2] != "Carries":
            continue  # role just shrank (run.py role_changes): results scatter more than projected
        best = {}
        for stat, x, pr in leg_menu(p):
            if stat == "pass" and p.get("new"):
                continue  # new starting QBs' passing has run ~20 yards high (see the PDF note)
            if MIN_LEG <= pr <= cap and (p["id"], stat) in levels and pr > best.get(stat, (0, 0, -1))[2]:
                best[stat] = (stat, x, pr)
        for stat, x, pr in best.values():
            bits, count = _bits(levels[(p["id"], stat)], pr)
            out.append(dict(id=p["id"], n=p["n"], t=p["t"], pos=p["p"], stat=stat, x=x, p=pr, bits=bits, count=count))
    return sorted(out, key=lambda c: -c["count"])


def best_parlay(cands, k, n, distinct=True):
    """The k legs that hit together in the most simulated games (exact search, pruned: a parlay only gets less
    likely as legs are added, and legs are tried likeliest first). distinct: at most one leg per player (two legs
    on one player, like his catches and his yards, are nearly the same bet and books price them that way)."""
    if len(cands) < k:
        return None
    best = [0, None]

    def dfs(start, chosen, bits):
        if len(chosen) == k:
            c = bits.bit_count()
            if c > best[0]:
                best[0], best[1] = c, list(chosen)
            return
        for i in range(start, len(cands) - (k - len(chosen)) + 1):
            if cands[i]["count"] <= best[0]:
                break  # this leg, and every later (less likely) one, can't beat the best parlay found
            if distinct and any(cands[j]["id"] == cands[i]["id"] for j in chosen):
                continue
            nb = bits & cands[i]["bits"]
            if nb.bit_count() <= best[0]:
                continue
            chosen.append(i)
            dfs(i + 1, chosen, nb)
            chosen.pop()

    dfs(0, [], (1 << (8 * ((n + 7) // 8))) - 1)  # every simulated game (as packed by _bits)
    if best[1] is None:
        return None
    legs = [cands[i] for i in best[1]]
    return dict(p=best[0] / n, ind=float(np.prod([c["p"] for c in legs])),
                legs=[dict(id=c["id"], n=c["n"], t=c["t"], pos=c["pos"], stat=c["stat"], x=c["x"], p=round(c["p"], 4),
                           label=leg_label(c["stat"], c["x"])) for c in legs])


# The 90% cap's parlays (2+ legs) hit ~5 points more often than predicted in both backtest seasons (2025 +4.9,
# 2026 +5.4; 70%/80% caps flip sign between seasons, so they stay as they are). A logit shift fit on either season
# improves the other's log loss (0.588 -> 0.581 / 0.583); pooled fit, 2026-11-12:
CAL_LOGIT = {0.9: 0.24}


def calibrate(p, cap, k):
    s = CAL_LOGIT.get(cap, 0.0) if k >= 2 else 0.0
    if not s:
        return p
    x = np.log(p / (1 - p)) + s
    return float(1 / (1 + np.exp(-x)))


def game_parlays(players, levels, n, calibrated=True):
    """{"k|cap": most likely k-leg parlay with every leg at most cap and one leg per player} for one game, plus
    "k|cap|same" allowing two or more legs on the same player. calibrated: apply CAL_LOGIT to the chance (the
    uncorrected one is kept as p_raw)."""
    out = {}
    for cap in CAPS:
        cands = candidates(players, levels, cap)
        for k in LEG_COUNTS:
            for distinct, suffix in ((True, ""), (False, "|same")):
                if k == 1 and not distinct:
                    continue
                r = best_parlay(cands, k, n, distinct)
                if r:
                    if calibrated and 0 < r["p"] < 1:
                        r["p_raw"], r["p"] = r["p"], calibrate(r["p"], cap, k)
                    out[f"{k}|{cap}{suffix}"] = r
    return out


def across_games(games, k, cap):
    """Most likely k-leg parlay with one leg per game: the k games whose likeliest leg under the cap is likeliest."""
    singles = [(g, g["parlays"][f"1|{cap}"]) for g in games if f"1|{cap}" in g.get("parlays", {})]
    singles.sort(key=lambda x: -x[1]["legs"][0]["p"])  # (stable: ties keep kickoff order, as on the page)
    if len(singles) < k:
        return None
    pick = singles[:k]
    legs = [dict(r["legs"][0], game=f'{g["away"]} at {g["home"]}') for g, r in pick]
    return dict(p=float(np.prod([lg["p"] for lg in legs])), legs=legs)


# ---------------------------------------------------------------------------------------------------------------
# Backtest: suggest parlays for past games exactly as run.py would, then check them against what happened
# ---------------------------------------------------------------------------------------------------------------
def backtest(seasons, gbm_files):
    import json

    import pandas as pd

    import simulate as S
    from blend import load_gbm
    from model import QUANTILES, qcol
    from touchdowns import p_any, p_two

    short = {"passing_yards": "pass", "rushing_yards": "rush", "receiving_yards": "recyd", "receptions": "rec"}
    pbp, feats, sched = (pd.read_parquet(f) for f in ("pbp.parquet", "features.parquet", "sched.parquet"))
    gbm = load_gbm(gbm_files)
    tdb = pd.read_parquet("td_backtest.parquet")
    cfg = json.load(open("blend.json"))
    qc = [qcol(q) for q in QUANTILES]
    rows = []
    for season in seasons:
        L = S.fit_league(pbp, list(range(season - 4, season)))
        T = S.tendency_tables(pbp, L)
        games = sched[(sched.season == season) & sched.home_score.notna() & sched.game_id.isin(feats.game_id)]
        for i, s in enumerate(games.itertuples()):
            f = feats[feats.game_id == s.game_id]
            G = S.game_params(f, s.home_team, s.away_team, s.gameday, T, L)
            htt = s.total_line / 2 + s.spread_line / 2
            sim, _ = S.simulate_game(G, L, S.N_SIMS, implied=(s.total_line - htt, htt), seed=i)
            sq = S.sim_quantiles(sim, QUANTILES)
            tdsim = S.sim_touchdowns(sim)
            pidx = {pid: j for j, pid in enumerate(G["pid"])}
            info = f.set_index("player_id")
            players = {}
            for r in gbm[gbm.game_id == s.game_id].itertuples():
                if r.player_id not in pidx or r.player_id not in info.index:
                    continue
                mk, j = short[r.market], pidx[r.player_id]
                b = cfg["blend"][r.market]
                q = (1 - b["sim_weight"]) * np.array([getattr(r, c) for c in qc]) + \
                    b["sim_weight"] * b["sim_scale"] * sq[mk][j]
                p = players.setdefault(r.player_id, dict(id=r.player_id, n=r.player_display_name, t=r.team,
                                                         p=r.position, m={}, f="",
                                                         new=bool(info.at[r.player_id, "qb_changed"] == 1 and
                                                                  info.at[r.player_id, "is_starting_qb"] == 1)))
                p["m"][mk] = list(np.sort(q))
            wt = cfg.get("td", {}).get("sim_weight", 0.5)
            for r in tdb[tdb.game_id == s.game_id].itertuples():
                if r.player_id not in pidx:
                    continue
                j = pidx[r.player_id]
                td = [(1 - wt) * p_any(r.lam) + wt * tdsim["any"][j], (1 - wt) * p_two(r.lam) + wt * tdsim["two"][j]]
                if r.player_id not in players and td[0] < 0.08:
                    continue  # run.py lists players without a yardage prop only from an 8% touchdown chance
                p = players.setdefault(r.player_id, dict(id=r.player_id, n=r.player_display_name, t=r.team,
                                                         p=r.position, m={}, f="", new=False))
                p["td"] = {"m": td}
            lv = sim_levels(sim, G)
            res = game_parlays(list(players.values()), lv, sim.n, calibrated=False)
            act = {pid: {"pass": a.passing_yards, "rush": a.rushing_yards, "recyd": a.receiving_yards,
                         "rec": a.receptions, "td": a.rushing_tds + a.receiving_tds}
                   for pid, a in info.iterrows()}
            for key, r in res.items():
                k, cap, *same = key.split("|")
                legs_hit = [bool(act[lg["id"]][lg["stat"]] >= lg["x"]) for lg in r["legs"]]
                hit = all(legs_hit)
                rows.append(dict(season=season, game_id=s.game_id, k=int(k), cap=float(cap), same=bool(same), p=r["p"],
                                 ind=r["ind"], hit=hit, legs_p=[lg["p"] for lg in r["legs"]], legs_hit=legs_hit))
            if (i + 1) % 50 == 0:
                print(f"  {season}: {i + 1}/{len(games)} games", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet("parlay_backtest.parquet")
    report(d)


def report(d):
    import pandas as pd
    for same, title in ((False, "one leg per player (default)"), (True, "same player allowed")):
        print(f"\nSuggested same-game parlays in past games, {title}:")
        print(f"{'legs':>4s} {'cap':>4s} {'games':>6s} {'predicted':>9s} {'hit':>6s} {'± se':>6s} {'if indep.':>9s}")
        for (k, cap), g in d[(d.k > 1) & (d.same == same)].groupby(["k", "cap"]):
            se = (g.hit.mean() * (1 - g.hit.mean()) / len(g)) ** 0.5
            print(f"{k:4d} {cap:4.0%} {len(g):6d} {g.p.mean():9.1%} {g.hit.mean():6.1%} {se:6.1%} {g.ind.mean():9.1%}")
    m = d[(d.k > 1) & ~d.same]
    br = lambda p: float(np.mean((p - m.hit.astype(float)) ** 2))  # noqa: E731
    print(f"\nBrier score of the parlay's chance: simulation-correlated {br(m.p):.4f}, legs treated as independent "
          f"{br(m.ind):.4f} (lower is better)")
    legs = pd.DataFrame({"p": np.concatenate(m.legs_p.to_numpy()), "hit": np.concatenate(m.legs_hit.to_numpy())})
    print(f"single legs used: predicted {legs.p.mean():.1%}, hit {legs.hit.mean():.1%} (n={len(legs)})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        import glob
        files = [a for a in sys.argv[2:] if a.endswith(".parquet")] or sorted(glob.glob("backtest_20[0-9][0-9].parquet"))
        backtest([int(a) for a in sys.argv[2:] if a.isdigit()] or [2025, 2026], files)
    else:
        raise SystemExit(__doc__)
