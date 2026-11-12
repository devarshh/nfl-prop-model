"""Walk-forward backtest of the game simulator: every game of the given seasons, using only what was known
before kickoff (league models fit on the 4 previous seasons, team and player tendencies from earlier games,
the closing betting lines).

    python backtest_sim.py 2025 2026      # writes sim_backtest.parquet and sim_td_backtest.parquet
                                          # (a few minutes per season)

Then python blend.py (yardage) and python touchdowns.py blend (touchdowns) compare it with the stats model's
backtest and pick how much weight each one gets.
"""
import sys
import time

import numpy as np
import pandas as pd

import simulate as S
from model import QUANTILES, qcol

MARKETS = {"pass": "passing_yards", "rush": "rushing_yards", "recyd": "receiving_yards", "rec": "receptions"}


def run_season(season, pbp, feats, sched, n_sims=S.N_SIMS):
    """Returns (yardage quantiles per player and prop, touchdown probabilities per player)."""
    L = S.fit_league(pbp, list(range(season - 4, season)))
    T = S.tendency_tables(pbp, L)
    games = sched[(sched.season == season) & sched.home_score.notna() & sched.game_id.isin(feats.game_id)]
    out, tds, t0 = [], [], time.time()
    for i, s in enumerate(games.itertuples()):
        rows = feats[feats.game_id == s.game_id]
        G = S.game_params(rows, s.home_team, s.away_team, s.gameday, T, L)
        implied = None
        if pd.notna(s.spread_line) and pd.notna(s.total_line):
            htt = s.total_line / 2 + s.spread_line / 2
            implied = (s.total_line - htt, htt)
        sim, k = S.simulate_game(G, L, n_sims, implied=implied, seed=i)
        q = S.sim_quantiles(sim, QUANTILES)
        for short, mk in MARKETS.items():
            col = S.STAT[short]
            mean = sim.stats[:, :, col].mean(0)
            df = pd.DataFrame(q[short], columns=[qcol(x) for x in QUANTILES])
            df.insert(0, "sim_mean", mean)
            df.insert(0, "market", mk)
            df.insert(0, "player_id", G["pid"])
            df.insert(0, "game_id", s.game_id)
            df["k"] = k[G["team"]]
            out.append(df)
        td = S.sim_touchdowns(sim)
        tds.append(pd.DataFrame({"game_id": s.game_id, "player_id": G["pid"], "p_any": td["any"],
                                 "p_two": td["two"], "p_first": td["first"], "td_mean": td["mean"],
                                 "team_td": np.asarray(sim.ev["td"].mean(0))[G["team"]]}))
        if (i + 1) % 50 == 0:
            print(f"  {season}: {i + 1}/{len(games)} games ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True), pd.concat(tds, ignore_index=True)


if __name__ == "__main__":
    seasons = [int(x) for x in sys.argv[1:]] or [int(pd.read_parquet("sched.parquet").season.max())]
    pbp = pd.read_parquet("pbp.parquet")
    feats = pd.read_parquet("features.parquet")
    sched = pd.read_parquet("sched.parquet")
    res = [run_season(s, pbp, feats, sched) for s in seasons]
    yds = pd.concat([r[0] for r in res], ignore_index=True)
    td = pd.concat([r[1] for r in res], ignore_index=True)
    yds.to_parquet("sim_backtest.parquet")
    td.to_parquet("sim_td_backtest.parquet")
    print(f"saved sim_backtest.parquet ({yds.game_id.nunique()} games, {len(yds)} player-props) "
          f"and sim_td_backtest.parquet ({len(td)} players)")
