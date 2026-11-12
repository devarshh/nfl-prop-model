"""
NFL player-prop projections in one command.

    python run.py                 # project the next unplayed week and open the page
    python run.py --week 10       # a specific week of the current season
    python run.py --no-download   # reuse data downloaded earlier (faster, but stale injuries)

Run it once mid-week to get early numbers, then again after Friday's injury reports.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import polars as pl
import nflreadpy as nfl

import parlays
import simulate as sim
from features import load_base, build_features, feature_list
from model import MARKETS, QUANTILES, market_masks, fit_quantiles, predict_quantiles, load_iters, qcol
from predict_week import future_rows, last_starters
from touchdowns import add_td_features, td_features, td_masks, fit_td, p_any, p_two

SHORT = {"passing_yards": "pass", "rushing_yards": "rush", "receiving_yards": "recyd", "receptions": "rec"}
STAT = {v: k for k, v in SHORT.items()}
TD_SHOW = 0.08  # players with no yardage prop are listed for touchdowns when their anytime chance is at least this


def step(msg):
    print(f"\n== {msg}", flush=True)


def download():
    sched = nfl.load_schedules()
    s = sched.to_pandas()
    upcoming = s.loc[s.home_score.isna() & (s.game_type == "REG"), "season"]
    season = int(upcoming.min()) if len(upcoming) else int(s.season.max())  # offseason: latest season
    years = list(range(2016, season + 1))
    nfl.load_player_stats(years).write_parquet("ps.parquet")
    nfl.load_snap_counts(years).write_parquet("sc.parquet")
    nfl.load_players().write_parquet("players.parquet")
    sched.write_parquet("sched.parquet")
    nfl.load_injuries(years).write_parquet("injuries.parquet")  # features.py: games players left hurt
    (nfl.load_ff_opportunity(years)
        .with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
        .write_parquet("ffopp.parquet"))
    sim.download_pbp(season)
    game_weather_history(season)


def game_weather_history(season):
    """game_weather.parquet: rain/snow flags per game from the play-by-play weather text (features.py: precip, snow)."""
    frames = [nfl.load_pbp([y]).select(["game_id", "weather", "roof"]).unique(subset=["game_id"])
              for y in range(2016, season + 1)]
    w = pl.concat(frames).to_pandas()
    t = w.weather.fillna("").str.lower()
    w["precip"] = t.str.contains(r"rain|shower|drizzle|snow|sleet|storm|flurr|wet|precip").astype(int)
    w["snow"] = t.str.contains(r"snow|flurr|sleet").astype(int)
    w.loc[w.roof.isin(["dome", "closed"]), ["precip", "snow"]] = 0
    w[["game_id", "weather", "precip", "snow"]].to_parquet("game_weather.parquet")


# Open-air stadiums (home team -> lat, lon) for the National Weather Service forecast
STADIUMS = {"BAL": (39.278, -76.623), "BUF": (42.774, -78.787), "CAR": (35.226, -80.853), "CHI": (41.862, -87.617),
            "CIN": (39.095, -84.516), "CLE": (41.506, -81.699), "DEN": (39.744, -105.020), "GB": (44.501, -88.062),
            "JAX": (30.324, -81.637), "KC": (39.049, -94.484), "MIA": (25.958, -80.239), "NE": (42.091, -71.264),
            "NYG": (40.813, -74.074), "NYJ": (40.813, -74.074), "PHI": (39.901, -75.168), "PIT": (40.447, -80.016),
            "SEA": (47.595, -122.332), "SF": (37.403, -121.970), "TB": (27.976, -82.503), "TEN": (36.166, -86.771),
            "WAS": (38.908, -76.864)}


def fetch_forecasts(sched, season, week):
    """forecast.json: temperature, wind and rain/snow for this week's open-air games from the National Weather
    Service hourly forecast (kickoff to +3 hours; only within ~a week of the game, US stadiums). Replaces the
    stadium's typical weather in features.py. Precip = 50%+ chance of rain or snow in those hours, on average."""
    import subprocess

    def get(u):  # curl uses the system's certificates (python.org Pythons on macOS often have none installed)
        r = subprocess.run(["curl", "-sf", "--max-time", "20", "-A", "nfl-prop-model (personal forecast script)",
                            "-H", "Accept: application/geo+json", u], capture_output=True, text=True, check=True)
        return json.loads(r.stdout)
    out = json.load(open("forecast.json")) if os.path.exists("forecast.json") else {}
    g = sched[(sched.season == season) & (sched.week == week) & sched.home_score.isna()]
    for _, r in g.iterrows():
        if r.roof != "outdoors" or r.location == "Neutral" or r.home_team not in STADIUMS:
            continue
        try:
            lat, lon = STADIUMS[r.home_team]
            hourly = get(get(f"https://api.weather.gov/points/{lat},{lon}")["properties"]["forecastHourly"])
            ko = pd.Timestamp(f"{r.gameday} {r.gametime}").tz_localize("America/New_York")
            hrs = [p for p in hourly["properties"]["periods"]
                   if ko - pd.Timedelta("30min") <= pd.Timestamp(p["startTime"]) <= ko + pd.Timedelta("3h")]
            if not hrs:
                continue
            pop = np.mean([(p.get("probabilityOfPrecipitation") or {}).get("value") or 0 for p in hrs]) / 100
            txt = " ".join(p["shortForecast"].lower() for p in hrs)
            wind = np.mean([float(str(p["windSpeed"]).split()[0]) for p in hrs])
            out[r.game_id] = dict(temp=round(float(np.mean([p["temperature"] for p in hrs])), 1), wind=round(float(wind), 1),
                                  precip=int(pop >= 0.5 and any(w in txt for w in ("rain", "shower", "snow", "drizzle", "storm"))),
                                  snow=int(pop >= 0.5 and "snow" in txt), pop=round(float(pop), 2), text=hrs[0]["shortForecast"])
            print(f"  {r.away_team} at {r.home_team}: {out[r.game_id]['temp']:.0f}F, wind {wind:.0f} mph, "
                  f"{pop:.0%} chance of precipitation ({hrs[0]['shortForecast']})")
        except Exception as e:  # a convenience: never block a run on the weather service
            print(f"  (no forecast for {r.away_team} at {r.home_team}: {e})")
    json.dump(out, open("forecast.json", "w"), indent=1)


def pick_week(sched, week=None):
    up = sched[sched.home_score.isna() & (sched.game_type == "REG")].sort_values("gameday")
    if up.empty:
        raise SystemExit("No unplayed regular-season games in the schedule (offseason or playoffs).")
    season = int(up.season.iloc[0])
    return season, int(week or up.week.iloc[0])


def fill_missing_qbs(sched, ps, season, week):
    """If the schedule hasn't listed a starter yet, assume last week's starter. Returns QB changes."""
    last_id, last_name = last_starters(ps, season)
    m = (sched.season == season) & (sched.week == week)
    changes = []
    for i in sched[m].index:
        for side in ("away", "home"):
            team = sched.at[i, f"{side}_team"]
            if pd.isna(sched.at[i, f"{side}_qb_id"]) and team in last_id:
                sched.at[i, f"{side}_qb_id"] = last_id[team]
                sched.at[i, f"{side}_qb_name"] = last_name[team]
            elif team in last_id and sched.at[i, f"{side}_qb_id"] != last_id[team]:
                changes.append((team, last_name[team], sched.at[i, f"{side}_qb_name"]))
    return sched, changes


def override_qbs(sched, ps, season, week, overrides):
    """--qb TEAM="Full Name": set a team's starting QB for this week when the nflverse schedule is stale
    (e.g. the listed starter was ruled Out). The QB must have a stat line for the team this season."""
    m = (sched.season == season) & (sched.week == week)
    for item in overrides or []:
        team, _, name = item.partition("=")
        team, name = team.strip().upper(), name.strip()
        q = ps[(ps.season == season) & (ps.position == "QB") & (ps.team == team) &
               (ps.player_display_name.str.lower() == name.lower())]
        if q.empty:
            raise SystemExit(f"--qb {item}: no {season} {team} QB named {name!r} in the stats data")
        pid, name = q.player_id.iloc[0], q.player_display_name.iloc[0]
        for side in ("away", "home"):
            i = sched.index[m & (sched[f"{side}_team"] == team)]
            sched.loc[i, f"{side}_qb_id"] = pid
            sched.loc[i, f"{side}_qb_name"] = name
        print(f"  {team} starting QB set to {name}")
    return sched


def warn_out_qbs(sched, season, week):
    """The schedule's starter can lag the news: flag listed starters ruled Out/Doubtful (they'd get no
    projection, and their teammates would be projected with the wrong QB)."""
    inj = nfl.load_injuries([season]).to_pandas()
    inj = inj[(inj.week == week) & inj.report_status.isin(["Out", "Doubtful"])]
    out = dict(zip(inj.gsis_id, inj.report_status))
    g = sched[(sched.season == season) & (sched.week == week) & sched.home_score.isna()]
    for _, r in g.iterrows():
        for side in ("away", "home"):
            if r[f"{side}_qb_id"] in out:
                print(f"\n!! {r[f'{side}_team']}: listed starter {r[f'{side}_qb_name']} is "
                      f"{out[r[f'{side}_qb_id']]}. Rerun with --qb {r[f'{side}_team']}=\"<starter>\" "
                      f"(the team gets no QB projection otherwise).")


def train_and_predict(df, mk, iters):
    feats = feature_list(df, mk)
    tr, _ = market_masks(df, mk)
    a = tr & (df.is_future != 1)
    fut = df[df.is_future == 1]
    models = fit_quantiles(df.loc[a, feats], df.loc[a, mk], mk, iters)
    return predict_quantiles(models, fut[feats], mk).add_prefix(f"{mk}_")


def train_td(df):
    """Stats model's expected touchdowns (Poisson rate) for every upcoming player."""
    feats = td_features(df)
    tr, _ = td_masks(df)
    a = tr & (df.is_future != 1)
    fut = df[df.is_future == 1]
    model = fit_td(df.loc[a, feats], df.loc[a, "tds"])
    return pd.DataFrame({"td_lam": model.predict(fut[feats])}, index=fut.index)


# Rain or snow at kickoff (outdoor games 2016-26 by the play-by-play weather text): totals finished 3.2 points under the
# closing line (167 games, SE 1.0; dry games +0.2), and 2022-26 play-by-play shows 3.3 points lower pass rate over
# expected and 3.5 points lower completion % over expected. In the simulator's logit units (p(1-p) ~ 0.23):
RAIN = {"total": -3.2, "proe": -0.14, "cpoe": -0.15}


def simulate_week(df, sched, season):
    """Play every upcoming game N_SIMS times. Returns {(game_id, player_id): {prop: simulated quantiles}} and,
    per game, each player's rank in every simulated game (for the parlays: which legs hit together)."""
    pbp = pd.read_parquet("pbp.parquet")
    L = sim.fit_league(pbp, list(range(season - 4, season)))
    T = sim.tendency_tables(pbp, L)
    s_idx = sched.set_index("game_id")
    fc = json.load(open("forecast.json")) if os.path.exists("forecast.json") else {}
    out, levels = {}, {}
    for i, (gid, rows) in enumerate(df[df.is_future == 1].groupby("game_id")):
        s = s_idx.loc[gid]
        G = sim.game_params(rows, s.home_team, s.away_team, s.gameday, T, L)
        implied = None
        if pd.notna(s.spread_line) and pd.notna(s.total_line):
            htt = s.total_line / 2 + s.spread_line / 2
            implied = (s.total_line - htt, htt)
        wx = fc.get(gid, {})
        if wx.get("precip"):  # rain/snow forecast: fewer passes, worse completions, lower scoring than the line
            r = wx.get("pop", 1.0)
            G["proe"] = G["proe"] + RAIN["proe"] * r
            G["cpoe"] = G["cpoe"] + RAIN["cpoe"] * r
            if implied:
                f = 1 + RAIN["total"] * r / s.total_line
                implied = (implied[0] * f, implied[1] * f)
            print(f"  {s.away_team} at {s.home_team}: rain adjustment x{r:.2f} (implied total "
                  f"{s.total_line:.1f} -> {sum(implied):.1f})" if implied else "")
        res, _ = sim.simulate_game(G, L, sim.N_SIMS, implied=implied, seed=i)
        q = sim.sim_quantiles(res, QUANTILES)
        td = sim.sim_touchdowns(res)
        for j, pid in enumerate(G["pid"]):
            out[(gid, pid)] = {short: q[short][j] for short in q}
            out[(gid, pid)]["td"] = (td["any"][j], td["two"][j], td["first"][j])
        levels[gid] = parlays.sim_levels(res, G)
    return out, levels


def blend_in_sims(w, sims):
    """Final quantiles = (1 - weight) x main model + weight x (scale x simulator), per stat, with the weights and
    scale corrections blend.py chose on past seasons. Keeps both models' quantiles for the page and PDF."""
    full = json.load(open("blend.json")) if os.path.exists("blend.json") else {}
    cfg = full.get("blend", {})
    keys = list(zip(w.game_id, w.player_id))
    for mk in MARKETS:
        short = SHORT[mk]
        cols = [f"{mk}_{qcol(q)}" for q in QUANTILES]
        model_q = w[cols].to_numpy(float)
        sim_q = np.array([sims[k][short] if k in sims else [np.nan] * len(QUANTILES) for k in keys], dtype=float)
        weight, scale = cfg.get(mk, {}).get("sim_weight", 0.0), cfg.get(mk, {}).get("sim_scale", 1.0)
        sim_q = scale * sim_q
        ok = ~np.isnan(sim_q).any(axis=1)
        final = np.where(ok[:, None], (1 - weight) * model_q + weight * np.nan_to_num(sim_q), model_q)
        w[cols] = np.sort(final, axis=1)
        w[[f"{mk}_model_{qcol(q)}" for q in QUANTILES]] = model_q
        w[[f"{mk}_sim_{qcol(q)}" for q in QUANTILES]] = sim_q
    # touchdowns: mix the two methods' probabilities (anytime and 2+); the first-TD chance is the simulation's
    wt = full.get("td", {}).get("sim_weight", 0.0)
    st = np.array([sims[k]["td"] if k in sims else (np.nan,) * 3 for k in keys], dtype=float)
    lam = w["td_lam"].to_numpy(float)
    w["td_model_any"], w["td_model_two"] = p_any(lam), p_two(lam)
    w["td_sim_any"], w["td_sim_two"], w["td_sim_first"] = st[:, 0], st[:, 1], st[:, 2]
    ok = ~np.isnan(st[:, 0])
    w["td_any"] = np.where(ok, (1 - wt) * w.td_model_any + wt * np.nan_to_num(st[:, 0]), w.td_model_any)
    w["td_two"] = np.where(ok, (1 - wt) * w.td_model_two + wt * np.nan_to_num(st[:, 1]), w.td_model_two)
    return w


ROLE_MIN_DIFF, ROLE_MIN_RATIO = 0.15, 1.4  # snap-share change that counts as a new role


# Role measures checked, with the change that counts (points, and 40%+ either way). Snap share alone missed backfield
# splits where both backs keep playing snaps but the carries move (MIN week 9: Mason took Jones's carries).
ROLE_MEASURES = {"Snaps": ("offense_pct", 0.15), "Carries": ("carry_share", 0.15), "Targets": ("target_share", 0.07)}


def role_changes(df, season):
    """{player_id: [share before, share in his last 2 games, measure]} for upcoming players whose role moved: snap
    share, carry share or target share changed by the ROLE_MEASURES points and 40%+ either way vs. his earlier games
    this season (3+ of them). Games he left hurt don't count. A drop wins over a rise; among drops the biggest.
    Also tags a player whose same-position teammate is back from a 2+ game absence (last 2 games or this week) when
    he got noticeably more of the ball while that teammate was out ("Teammate back"). The models' usage averages
    still lean on the old role for a few games, so the page flags these and caps the overs."""
    fut = df[df.is_future == 1]
    h = df[(df.season == season) & (df.is_future != 1) & (df.injury_exit != 1) & df.offense_pct.notna()]
    h = h[h.player_id.isin(fut.player_id)].sort_values("seq")
    out = {}
    for pid, g in h.groupby("player_id"):
        if len(g) < 5:
            continue
        best = None
        for name, (col, diff) in ROLE_MEASURES.items():
            x = g[col].fillna(0)
            before, recent = x.iloc[:-2].mean(), x.iloc[-2:].mean()
            if name != "Snaps" and before < 0.08 and recent < 0.08:
                continue  # a bit player's share: noise
            ratio = max(before, recent) / max(min(before, recent), 0.01)
            if abs(recent - before) >= diff and ratio >= ROLE_MIN_RATIO:
                score = (recent < before, abs(recent - before) / diff)
                if best is None or score > best[0]:
                    best = (score, [round(float(before), 2), round(float(recent), 2), name])
        if best:
            out[pid] = best[1]

    # teammate back: same team and position, missed 2+ team games this season, played in the team's last 2 or is
    # projected this week after missing the last one
    s = df[(df.season == season) & (df.is_future != 1)]
    team_games = s.groupby("team").seq.agg(lambda x: sorted(set(x))).to_dict()
    played = s[s.offense_pct.fillna(0) > 0].groupby("player_id").seq.agg(set).to_dict()
    names = df.drop_duplicates("player_id").set_index("player_id").player_display_name
    for _, r in fut.iterrows():
        if r.player_id in out or r.position not in ("RB", "WR", "TE"):
            continue
        tg = team_games.get(r.team, [])
        mine = played.get(r.player_id, set())
        col = "carry_share" if r.position == "RB" else "target_share"
        for _, m in fut[(fut.team == r.team) & (fut.position == r.position) & (fut.player_id != r.player_id)].iterrows():
            his = played.get(m.player_id, set())
            missed = [x for x in tg if x not in his]
            back = len(missed) >= 2 and (any(x in his for x in tg[-2:]) or (tg and tg[-1] not in his))
            hx = s[(s.player_id == m.player_id) & s.seq.isin(his)]
            regular = (len(hx) >= 2 and hx.offense_pct.median() >= 0.4 and
                       hx[col].fillna(0).mean() >= (0.25 if col == "carry_share" else 0.12))
            if not (back and regular):
                continue
            gx = s[(s.player_id == r.player_id) & s.seq.isin(mine)]
            wo, wi = gx[gx.seq.isin(missed)][col].fillna(0), gx[~gx.seq.isin(missed)][col].fillna(0)
            without, with_ = wo.mean(), wi.mean()
            if len(wo) >= 2 and len(wi) >= 2 and without - with_ >= 0.08 and without >= 1.25 * with_:
                out[r.player_id] = [round(float(without), 2), round(float(with_), 2), f"Teammate back: {names[m.player_id]}"]
                break
    return out


def build_games(w, sched, roles=None):
    """Upcoming games with every player's selected props (quantiles at the QUANTILES levels)."""
    pos = w.position
    sel = {
        "passing_yards": w.is_starting_qb == 1,
        "rushing_yards": (pos.isin(["RB", "FB"]) & ((w.carries_ewm >= 6) | (w.rushing_yards_q50 >= 20))) |
                         ((w.is_starting_qb == 1) & ((w.rushing_yards_ewm >= 15) | (w.rushing_yards_q50 >= 15))),
        "receiving_yards": pos.isin(["WR", "TE", "RB"]) & ((w.targets_ewm >= 3.5) | (w.receiving_yards_q50 >= 20)),
    }
    sel["receptions"] = sel["receiving_yards"]
    td_ok = pos.isin(["RB", "FB", "WR", "TE"]) | ((pos == "QB") & (w.is_starting_qb == 1))
    s_idx = sched.set_index("game_id")
    order = {"QB": 0, "RB": 1, "FB": 1, "WR": 2, "TE": 3}
    mid = QUANTILES.index(0.5)
    games = []
    for gid, g in w.groupby("game_id", sort=False):
        s = s_idx.loc[gid]
        players = []
        for _, r in g.iterrows():
            props = [mk for mk in MARKETS if sel[mk].loc[r.name]]
            qs = lambda kind, mk: [round(float(r[f"{mk}_{kind}{qcol(q)}"]), 2) for q in QUANTILES]  # noqa: E731
            m = {SHORT[mk]: qs("", mk) for mk in props}                      # blended: what the page uses
            gm = {SHORT[mk]: qs("model_", mk) for mk in props}               # main model on its own
            sm = {SHORT[mk]: qs("sim_", mk) for mk in props if pd.notna(r[f"{mk}_sim_q50"])}  # simulator
            td = None
            if td_ok.loc[r.name] and pd.notna(r.td_any):  # anytime and 2+ TDs: blend, stats model, simulation
                r3 = lambda *v: [round(float(x), 3) for x in v]  # noqa: E731
                td = dict(m=r3(r.td_any, r.td_two), g=r3(r.td_model_any, r.td_model_two))
                if pd.notna(r.td_sim_any):
                    td.update(s=r3(r.td_sim_any, r.td_sim_two), first=round(float(r.td_sim_first), 3))
            if m or (td and td["m"][0] >= TD_SHOW):
                rc = (roles or {}).get(r.player_id)
                players.append(dict(id=r.player_id, n=r.player_display_name, t=r.team, p=r.position,
                                    f=r.flag or "", new=bool(r.is_starting_qb == 1 and r.qb_changed == 1),
                                    m=m, g=gm, s=sm, **({"td": td} if td else {}), **({"rc": rc} if rc else {})))
        players.sort(key=lambda x: (x["t"] != s.away_team, order[x["p"]],
                                    -max([v[mid] for v in x["m"].values()] or [0])))
        has_line = pd.notna(s.spread_line) and pd.notna(s.total_line)  # lines can be missing early in the week
        home_tt = s.total_line / 2 + s.spread_line / 2 if has_line else None
        games.append(dict(id=gid, away=s.away_team, home=s.home_team, day=s.gameday,
                          time=s.gametime if pd.notna(s.gametime) else "",
                          spread=float(s.spread_line) if has_line else None,
                          total=float(s.total_line) if has_line else None,
                          away_tt=round(s.total_line - home_tt, 1) if has_line else None,
                          home_tt=round(home_tt, 1) if has_line else None,
                          away_qb=s.away_qb_name if pd.notna(s.away_qb_name) else "TBD",
                          home_qb=s.home_qb_name if pd.notna(s.home_qb_name) else "TBD",
                          stadium=s.stadium if pd.notna(s.stadium) else "",
                          neutral=bool(s.location == "Neutral"), players=players))
    return games


def add_started_games(games, season, week, sched, ps):
    """Carry over this week's games that have kicked off since the last run, keeping their last pre-game
    projections. Once a game is final, attach the score and each player's actual stats (a = None: no stat line,
    i.e. he didn't play) so make_pdf.py can show how the projections did."""
    path = f"output/week{week}.json"
    if not os.path.exists(path):
        return games
    upcoming = {g["id"] for g in games}
    started = [g for g in json.load(open(path))
               if g["id"].startswith(f"{season}_{week:02d}_") and g["id"] not in upcoming]
    s_idx = sched.set_index("game_id")
    for g in started:
        g["started"] = True
        s = s_idx.loc[g["id"]]
        if pd.isna(s.home_score):
            continue  # in progress, or the final score isn't in the data yet
        g["final"] = {"away": int(s.away_score), "home": int(s.home_score)}
        box = ps[ps.game_id == g["id"]]
        by_id = {r.player_id: r for r in box.itertuples()}
        by_name = {(r.player_display_name, r.team): r for r in box.itertuples()}
        for p in g["players"]:
            r = by_id.get(p.get("id")) or by_name.get((p["n"], p["t"]))
            p["a"] = {k: float(getattr(r, STAT[k])) for k in p["m"]} if r is not None else None
            if r is not None and "td" in p:
                p["a"]["td"] = float(r.rushing_tds + r.receiving_tds)
    return sorted(games + started, key=lambda x: (x["day"], x["time"]))


def write_outputs(games, season, week):
    """output/weekN.json keeps every game of the week; the interactive page shows only games still to come."""
    os.makedirs("output", exist_ok=True)
    json.dump(games, open(f"output/week{week}.json", "w"))
    upcoming = [g for g in games if not g.get("started")]
    run = dt.datetime.now().strftime("%A, %b %-d at %-I:%M %p") if os.name != "nt" else dt.datetime.now().strftime("%A, %b %d")
    html = (open("week_template.html").read().replace("/*DATA*/", json.dumps(upcoming))
            .replace("/*QS*/", json.dumps(QUANTILES))
            .replace("/*WEEK*/", str(week)).replace("/*SEASON*/", str(season)).replace("/*RUN*/", run))
    path = os.path.abspath(f"output/week{week}_projections.html")
    open(path, "w").write(html)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--qb", action="append", metavar='TEAM="Name"',
                    help="override a team's starting QB this week (repeatable)")
    ap.add_argument("--out", action="append", metavar='"Name"',
                    help="a player ruled out or inactive after the injury report (repeatable); his targets and "
                         "carries go to the teammates who play")
    args = ap.parse_args()
    t0 = time.time()

    if not args.no_download:
        step("Downloading nflverse data")
        download()
    sched = pd.read_parquet("sched.parquet")
    ps = pd.read_parquet("ps.parquet")
    season, week = pick_week(sched, args.week)
    print(f"Projecting {season} week {week}")

    sched = override_qbs(sched, ps, season, week, args.qb)
    sched, changes = fill_missing_qbs(sched, ps, season, week)
    sched.to_parquet("sched.parquet")
    try:
        warn_out_qbs(sched, season, week)
    except Exception as e:  # the injury feed is a convenience check; never block a run on it
        print(f"(couldn't check the injury report for QBs: {e})")
    if changes:
        print("\nQB changes from last week (double-check these in the news):")
        for team, old, new in changes:
            print(f"  {team}: {old} -> {new}")

    step("Weather forecasts for open-air games")
    try:
        fetch_forecasts(sched, season, week)
    except Exception as e:
        print(f"(couldn't fetch forecasts: {e})")

    step("Building the expected-to-play list")
    fr, flags = future_rows(season, week, sched, ps)
    for name in args.out or []:
        hit = fr.player_display_name.str.lower() == name.strip().lower()
        if not hit.any():
            raise SystemExit(f"--out {name!r}: no player by that name in this week's expected-to-play list")
        print(f"  left out: {', '.join(fr.loc[hit, 'player_display_name'] + ' (' + fr.loc[hit, 'team'] + ')')}")
        fr = fr[~hit]
    if fr.empty:
        if os.path.exists(f"output/week{week}.json"):  # still record the latest results for make_pdf.py
            write_outputs(add_started_games([], season, week, sched, ps), season, week)
        raise SystemExit(f"Every week-{week} game has kicked off (results saved to output/week{week}.json). "
                         f"Use --week {week + 1} for the next one.")
    print(f"{len(fr)} players across {fr.game_id.nunique()} games")

    step("Building features")
    df = add_td_features(build_features(load_base(extra_rows=fr)))
    df["flag"] = df.player_id.map(flags).where(df.is_future == 1, "")

    w = df[df.is_future == 1].copy()
    iters = load_iters()
    for mk in MARKETS:
        step(f"Training {mk} models")
        w = w.join(train_and_predict(df, mk, iters[mk]))
    step("Training the touchdown model")
    w = w.join(train_td(df))

    step(f"Simulating every game {sim.N_SIMS:,} times")
    sims, levels = simulate_week(df, sched, season)
    w = blend_in_sims(w, sims)

    step("Finding the most likely parlays")
    roles = role_changes(df, season)
    if roles:
        names = df.drop_duplicates("player_id").set_index("player_id").player_display_name
        print("Role changes (earlier games -> last 2):",
              "; ".join(f"{names[p]} {k if k.startswith('Teammate') else k + ' ' + format(a, '.0%') + '->' + format(b, '.0%')}"
                        for p, (a, b, k) in roles.items()))
    new_games = build_games(w, sched, roles)
    for g in new_games:
        g["parlays"] = parlays.game_parlays(g["players"], levels[g["id"]], sim.N_SIMS)

    step("Building the page")
    games = add_started_games(new_games, season, week, sched, ps)
    path = write_outputs(games, season, week)
    upcoming = [g for g in games if not g.get("started")]
    n = sum(len(p["m"]) for g in upcoming for p in g["players"])
    n_td = sum("td" in p for g in upcoming for p in g["players"])
    finals = [f'{g["away"]} {g["final"]["away"]}, {g["home"]} {g["final"]["home"]}' for g in games if g.get("final")]
    print(f"{len(upcoming)} upcoming games, {n} props, {n_td} touchdown projections. Saved to {path}  "
          f"({time.time() - t0:.0f}s)")
    if finals:
        print(f"Final, with results kept for make_pdf.py: {'; '.join(finals)}")
    if not args.no_open:
        webbrowser.open("file://" + path)


if __name__ == "__main__":
    main()
