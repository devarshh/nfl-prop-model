"""
Monte Carlo game simulator: plays every snap of a game, 1,000 times.

League behaviour comes from nflverse play-by-play (the last few complete seasons):
  - play calling: pass or run by down, distance, field position, score and clock (a gradient-boosted model,
    tabulated so each snap is a lookup); 4th downs: go / punt / field goal the same way
  - pass plays: sacks, scrambles, who gets targeted, how deep (air yards), completion and interception odds by
    depth and field position, yards after the catch
  - runs, punts, kickoffs (current rules), field goals by distance, extra points and 2-point tries, penalties,
    fumbles, interception returns, and how much clock each kind of play takes in each game situation
Each team brings its own tendencies, measured from its recent games and shrunk toward the league when the
sample is small: how much more or less it passes than the situation calls for, tempo, 4th-down aggressiveness,
its QB's sack, scramble, accuracy and interception habits, and its defense's pass rush, coverage and run defense.
Each player brings his usage (share of targets and carries among the players active that day), depth of target,
catch rate and yards after the catch over expected, and yards per carry.
Finally every game is tuned so its average simulated score matches the betting market's implied team totals.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

N_SIMS = 1000
PBP_COLS = ["game_id", "season", "week", "season_type", "game_date", "home_team", "away_team", "posteam", "defteam",
            "play_id", "drive", "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining",
            "half_seconds_remaining", "score_differential", "posteam_score", "defteam_score", "play_type", "pass",
            "rush", "qb_dropback", "qb_scramble", "sack", "pass_attempt", "complete_pass", "interception",
            "air_yards", "yards_after_catch", "yards_gained", "xpass", "cp", "cpoe", "xyac_mean_yardage",
            "passer_player_id", "receiver_player_id", "rusher_player_id", "touchdown", "pass_touchdown",
            "rush_touchdown", "return_touchdown", "fumble_lost", "safety", "field_goal_result", "kick_distance",
            "extra_point_result", "two_point_attempt", "two_point_conv_result", "punt_blocked", "touchback",
            "return_yards", "penalty", "penalty_team", "penalty_yards", "first_down_penalty", "no_huddle",
            "qb_kneel", "qb_spike", "timeout", "out_of_bounds", "first_down", "fourth_down_converted",
            "fourth_down_failed", "home_score", "away_score", "spread_line", "total_line", "roof",
            "kickoff_attempt", "punt_attempt", "field_goal_attempt"]


def download_pbp(season, path="pbp.parquet"):
    """Play-by-play for the four seasons the league models use plus the current one."""
    import nflreadpy as nfl
    nfl.load_pbp(list(range(season - 4, season + 1))).select(PBP_COLS).write_parquet(path)
STAT = {"pass": 0, "rush": 3, "recyd": 5, "rec": 6}  # columns of the per-player stat array used for props
NSTAT = 10  # pass_yds, pass_att, pass_cmp, rush_yds, carries, rec_yds, rec, targets, rush_td, rec_td
TD_RUSH, TD_REC = 8, 9
KO, SCRIM, PAT = 0, 1, 2

# ---- bins for the play-calling lookup tables (finer where decisions change fastest)
TOGO_EDGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 21]
TOGO_REP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.5, 14.5, 18, 24]
YL_EDGES = [1, 2, 3, 4, 5, 6, 8, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96]
SD_EDGES = [-99, -21, -17, -14, -13, -10, -8, -7, -6, -3, -2, 0, 1, 3, 4, 7, 8, 9, 11, 14, 15, 18, 22]
SD_REP = [-25, -19.5, -16, -14, -12, -9.5, -8, -7, -5, -3, -1.5, 0, 1.5, 3, 5, 7, 8, 9.5, 12, 14, 16, 19.5, 25]
HSR_EDGES = [0, 20, 40, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1500]  # seconds left in the half
ADOT_EDGES = [-99, 2, 5, 7.5, 10, 12.5, 15]   # receiver depth-of-target groups (air-yard shapes differ)
AIR_EDGES = [-99, 1, 3, 6, 10, 15, 20]        # yards-after-catch shapes by depth of the catch
PUNT_EDGES = list(range(0, 100, 5))
RZ_YL_EDGES = [1, 3, 6, 11, 16, 21]           # inside the 25 routes compress: air yards by field position...
RZ_TYPE_EDGES = [-99, 5, 10]                  # ...and receiver type (short, intermediate, deep)
GOAL_YL_EDGES = [1, 2, 3, 4, 6, 8]            # runs inside the 10 by exact distance (the end zone caps the gain)
RUN_YL_EDGES = [11, 16, 21, 30, 50, 70]       # ...and outside it by field position: near the goal line defenses
                                              # squeeze runs, deep in a team's own end long runs have more room
# Yards after the catch for a catch with this much room in front of it (up to the next edge) come from real catches
# that had at least the same room, so the goal line never cut them short: 25+ yards for any catch with less room,
# 40+ for 25-39, 60+ for 40-59, 80+ beyond. (Catches with less room would understate long touchdowns.)
YAC_ROOM = [25, 40, 60, 80]
NO_SCALE_RUN, NO_SCALE_PASS = 20, 10  # inside these yard lines gains are the league's real outcomes, unscaled
SITU = ["down", "ydstogo", "yardline_100", "score_differential", "half_seconds_remaining", "game_seconds_remaining"]


def _edges_map(edges, lo, hi):
    """Array mapping every integer value lo..hi to its bin index."""
    v = np.arange(lo, hi + 1)
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 1)


TOGO_MAP = _edges_map(TOGO_EDGES, 0, 99)
YL_MAP = _edges_map(YL_EDGES, 0, 99)
SD_MAP = _edges_map(SD_EDGES, -60, 60)          # index with sd + 60
HSR_MAP = _edges_map(HSR_EDGES, 0, 1800)
YL_REP = [(a + b - 1) / 2 for a, b in zip(YL_EDGES, YL_EDGES[1:] + [100])]
HSR_REP = [(a + b) / 2 for a, b in zip(HSR_EDGES, HSR_EDGES[1:] + [1800])]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _gbm(X, y, rounds, **params):
    base = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, verbose=-1, num_threads=4)
    return lgb.train(dict(base, **params), lgb.Dataset(np.asarray(X, dtype=float), np.asarray(y, dtype=float)),
                     num_boost_round=rounds)


def _binned(values, bins, nbins):
    """Empirical distribution per bin as (sorted values, start, count) for fast vectorized sampling."""
    values, bins = np.asarray(values, dtype=float), np.asarray(bins)
    order = np.argsort(bins, kind="stable")
    v, b = values[order], bins[order]
    count = np.bincount(b, minlength=nbins)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    for i in np.flatnonzero(count == 0):  # an empty bin borrows its nearest neighbour
        j = np.flatnonzero(count)[np.argmin(np.abs(np.flatnonzero(count) - i))]
        start[i], count[i] = start[j], count[j]
    return v, start, count


def _draw(dist, bins, rng):
    v, start, count = dist
    return v[start[bins] + (rng.random(len(bins)) * count[bins]).astype(int)]


def _draw1(arr, n, rng):
    return arr[rng.integers(0, len(arr), n)]


def _next_snap_yardline(p, rows, team_col):
    """yardline_100 of the next snap by the team in team_col after each special-teams play (NaN if none)."""
    snaps = p[p.play_type.isin(["pass", "run"]) & p.down.notna()][["game_id", "play_id", "posteam", "yardline_100"]]
    nxt = snaps.rename(columns={"play_id": "next_play", "yardline_100": "next_yl"}).sort_values("next_play")
    r = rows[["game_id", "play_id", team_col]].rename(columns={team_col: "posteam"})
    r = r.assign(row=np.arange(len(r))).sort_values("play_id")
    m = pd.merge_asof(r, nxt, left_on="play_id", right_on="next_play", by=["game_id", "posteam"],
                      direction="forward", allow_exact_matches=False)
    return m.sort_values("row")["next_yl"].to_numpy(dtype=float)


def _clock_rows(p):
    """Every snap followed by another snap in the same drive and half: seconds until that next snap, whether the
    clock stopped, and the game situation."""
    allsn = p[p.down.notna() & (p.play_type.isin(["pass", "run"]) | (p.play_type.eq("no_play") & p.penalty.eq(1)))]
    sn = allsn.sort_values(["game_id", "play_id"])
    nx = sn.groupby("game_id")[["game_seconds_remaining", "drive", "qtr"]].shift(-1)
    same = (nx.drive == sn.drive) & ((nx.qtr <= 2) == (sn.qtr <= 2)) & (sn.qtr <= 4) & (nx.qtr <= 4)
    s2 = sn[same].copy()
    s2["elapsed"] = (sn.game_seconds_remaining - nx.game_seconds_remaining)[same]
    s2 = s2[(s2.elapsed >= 0) & (s2.elapsed <= 60)]
    # out-of-bounds plays only stop the clock late in halves; that is already in each situation's mix of times
    s2["stop"] = (((s2.pass_attempt == 1) & (s2.complete_pass == 0) & (s2.interception == 0) & (s2.sack == 0))
                  | (s2.play_type == "no_play")).astype(int)
    s2["sit"] = clock_situation((s2.qtr >= 3).astype(int).to_numpy(), s2.half_seconds_remaining.to_numpy(),
                                s2.score_differential.to_numpy())
    return s2


# =====================================================================================================
# League models
# =====================================================================================================
def fit_league(pbp, seasons):
    """Everything about how NFL games are played, learned from the given seasons of play-by-play."""
    p = pbp[pbp.season.isin(seasons)].copy()
    L = {"seasons": list(seasons)}
    snaps = p[p.play_type.isin(["pass", "run"]) & p.down.notna()]

    # --- pass/run calling and 4th downs, tabulated over binned situations
    xp = _gbm(snaps[SITU], snaps.qb_dropback, 300, objective="binary", num_leaves=63, min_data_in_leaf=200)
    d, t, y, s, h, hs = np.meshgrid(np.arange(1, 5), TOGO_REP, YL_REP, SD_REP, [0, 1], HSR_REP, indexing="ij")
    gsr = np.where(h == 0, 1800 + hs, hs)
    X = np.column_stack([a.ravel() for a in (d, t, y, s, hs, gsr)])
    L["xpass"] = xp.predict(X, raw_score=True).reshape(d.shape).astype(np.float32)

    f4 = p[(p.down == 4) & p.play_type.isin(["pass", "run", "punt", "field_goal"])]
    cls = np.select([f4.play_type.isin(["pass", "run"]), f4.play_type.eq("punt")], [0, 1], 2)
    c4 = ["ydstogo", "yardline_100", "score_differential", "half_seconds_remaining", "game_seconds_remaining"]
    m4 = _gbm(f4[c4], cls, 250, objective="multiclass", num_class=3, min_data_in_leaf=60)
    t, y, s, h, hs = np.meshgrid(TOGO_REP, YL_REP, SD_REP, [0, 1], HSR_REP, indexing="ij")
    gsr = np.where(h == 0, 1800 + hs, hs)
    X = np.column_stack([a.ravel() for a in (t, y, s, hs, gsr)])
    L["fourth"] = m4.predict(X, raw_score=True).reshape(t.shape + (3,)).astype(np.float32)
    L["fourth_model"], L["fourth_cols"] = m4, c4

    # --- passing: completion and interception odds by depth of target and field position
    att = p[(p.play_type == "pass") & (p.pass_attempt == 1) & (p.sack != 1) & p.air_yards.notna()]
    air = np.arange(-15, 71)
    G = np.array([(a, yl) for a in air for yl in range(1, 100)], dtype=float)
    cpm = _gbm(att[["air_yards", "yardline_100"]], att.complete_pass, 200, objective="binary", min_data_in_leaf=300)
    intm = _gbm(att[["air_yards", "yardline_100"]], att.interception, 150, objective="binary", min_data_in_leaf=1500,
                num_leaves=15)
    L["cp"] = cpm.predict(G, raw_score=True).reshape(len(air), 99).astype(np.float32)
    L["int"] = intm.predict(G, raw_score=True).reshape(len(air), 99).astype(np.float32)

    # air yards around each receiver's own depth of target, by receiver type
    tg = att[att.receiver_player_id.notna()]
    rs = tg.groupby(["season", "receiver_player_id"]).air_yards.agg(["mean", "size"])
    tg = tg.join(rs[rs["size"] >= 20], on=["season", "receiver_player_id"], how="inner")
    L["air_resid"] = _binned(tg.air_yards - tg["mean"], np.searchsorted(ADOT_EDGES, tg["mean"], "right") - 1,
                             len(ADOT_EDGES))
    rz = tg[tg.yardline_100 <= 25]
    cell = (np.searchsorted(RZ_YL_EDGES, rz.yardline_100, "right") - 1) * 3 + \
           (np.searchsorted(RZ_TYPE_EDGES, rz["mean"], "right") - 1)
    L["air_rz"] = _binned(rz.air_yards, cell, len(RZ_YL_EDGES) * 3)
    cmp_ = att[(att.complete_pass == 1) & att.yards_after_catch.notna()]
    room, abin = cmp_.yardline_100 - cmp_.air_yards, np.searchsorted(AIR_EDGES, cmp_.air_yards, "right") - 1
    vals, cells = [], []
    for b, need in enumerate(YAC_ROOM):
        k = (room >= need).to_numpy()
        vals.append(cmp_.yards_after_catch.to_numpy(float)[k])
        cells.append(abin[k] * len(YAC_ROOM) + b)
    L["yac"] = _binned(np.concatenate(vals), np.concatenate(cells), len(AIR_EDGES) * len(YAC_ROOM))

    # --- runs (designed), scrambles, sacks, fumbles
    run = p[(p.play_type == "run") & (p.qb_scramble != 1) & p.down.notna() & p.rusher_player_id.notna()]
    yl_r, togo_r, gain_r = run.yardline_100.to_numpy(), run.ydstogo.to_numpy(), run.yards_gained.to_numpy(float)
    b = rush_bin(yl_r, togo_r)
    keep = ~np.isin(b, list(RUN_FAR))
    vals, cells = [gain_r[keep]], [b[keep]]
    for cell, need in RUN_FAR.items():  # open field: runs that started farther out, never cut short by the end zone
        k = (yl_r >= need) & (togo_r > 2)
        vals.append(gain_r[k])
        cells.append(np.full(k.sum(), cell))
    passers = p.loc[p.pass_attempt == 1, "passer_player_id"].value_counts()
    qbr = (run.rusher_player_id.isin(passers[passers >= 50].index).to_numpy()) & (yl_r <= 20)
    vals.append(gain_r[qbr])
    cells.append(qb_rush_bin(yl_r[qbr]))
    L["rush"] = _binned(np.concatenate(vals), np.concatenate(cells), N_RUN_DIST)
    scr = p[(p.play_type == "run") & (p.qb_scramble == 1) & (p.yardline_100 >= 30)]
    L["scramble"] = scr.yards_gained.to_numpy(float)
    sk = p[p.sack == 1]
    L["sack"] = sk.yards_gained.to_numpy(float)
    L["fum_run"] = run.fumble_lost.mean()
    L["fum_rec"] = att[att.complete_pass == 1].fumble_lost.mean()
    L["fum_sack"] = sk.fumble_lost.mean()
    ints = p[p.interception == 1]
    L["int_return"] = ints.return_yards.fillna(0).to_numpy(float)

    # league averages used as priors when shrinking team and player numbers
    db = snaps[snaps.qb_dropback == 1]
    # (ypc: all designed runs, the same basis as each runner's history; mid-field runs alone average more)
    L["lg"] = dict(sack=db.sack.mean(), scramble=db.qb_scramble.mean(), int=att.interception.mean(),
                   cmp=att.complete_pass.mean(), ypc=run.yards_gained.mean(), air=att.air_yards.mean(),
                   scr_ypc=L["scramble"].mean(), ypc_pass=att[att.complete_pass == 1].yards_gained.mean(),
                   pass_rate=snaps.qb_dropback.mean())
    # nflverse's expected completions / yards after catch don't average exactly to what happened in these
    # seasons; measure players against the league instead
    a = att[att.cp.notna()]
    L["lg"]["cpoe"] = float((a.complete_pass - a.cp).mean())
    c = att[(att.complete_pass == 1) & att.yards_after_catch.notna() & att.xyac_mean_yardage.notna()]
    L["lg"]["yac_ratio"] = float(c.yards_after_catch.sum() / c.xyac_mean_yardage.sum())

    # --- special teams
    pu = p[(p.play_type == "punt")].copy()
    nyl = _next_snap_yardline(p, pu, "defteam")
    nyl = np.where(pu.return_touchdown.eq(1), 0, nyl)
    ok = ~np.isnan(nyl)
    L["punt"] = _binned(nyl[ok], np.searchsorted(PUNT_EDGES, pu.yardline_100.to_numpy()[ok], "right") - 1,
                        len(PUNT_EDGES))
    last = max(seasons)  # kickoff rules changed in 2024 and 2025: use the latest season only
    ko = p[(p.play_type == "kickoff") & (p.season == last)].copy()
    nyl = _next_snap_yardline(p, ko, "posteam")
    nyl = np.where(ko.return_touchdown.eq(1), 0, nyl)
    L["kickoff"] = nyl[~np.isnan(nyl) & (nyl >= 0) & (nyl <= 99)]
    fg = p[(p.play_type == "field_goal") & p.kick_distance.notna()]
    fgm = _gbm(fg[["kick_distance"]], fg.field_goal_result.eq("made"), 150, objective="binary",
               monotone_constraints=[-1], min_data_in_leaf=50, num_leaves=7)
    L["fg"] = fgm.predict(np.arange(0, 81, dtype=float)[:, None])
    xp_ = p[p.extra_point_result.notna()]
    L["xp"] = xp_.extra_point_result.eq("good").mean()
    two = p[p.two_point_attempt == 1]
    L["two_rate"] = two.two_point_conv_result.eq("success").mean()
    tries = p[(p.extra_point_result.notna()) | (p.two_point_attempt == 1)]
    sdc = tries.score_differential.clip(-20, 20).astype(int) + 20
    late = tries.qtr >= 4
    go2 = np.zeros((2, 41))
    for k, mask in enumerate([~late, late]):
        n = np.bincount(sdc[mask], minlength=41)
        g2 = np.bincount(sdc[mask], weights=tries.two_point_attempt[mask].fillna(0), minlength=41)
        base = tries.two_point_attempt[mask].fillna(0).mean()
        go2[k] = (g2 + 10 * base) / (n + 10)
    L["go2"] = go2

    # --- penalties that wipe out the play (stats don't count)
    allsn = p[p.down.notna() & (p.play_type.isin(["pass", "run"]) | (p.play_type.eq("no_play") & p.penalty.eq(1)))]
    pen = allsn[allsn.play_type.eq("no_play")]
    off = pen[pen.penalty_team == pen.posteam]
    de = pen[pen.penalty_team == pen.defteam]
    L["pen_off"] = len(off) / len(allsn)
    L["pen_def"] = len(de) / len(allsn)
    # penalties accepted on plays that still count (stats stand; the ball moves and the defense's usually
    # gives a first down)
    live = allsn[allsn.play_type.isin(["pass", "run"]) & allsn.penalty.eq(1)]
    lde, lof = live[live.penalty_team == live.defteam], live[live.penalty_team == live.posteam]
    L["live_def"], L["live_off"] = len(lde) / len(allsn), len(lof) / len(allsn)
    L["live_def_yds"] = lde.penalty_yards.fillna(5).clip(0, 40).to_numpy(float)
    L["live_def_first"] = lde.first_down_penalty.fillna(0).to_numpy(float)
    L["live_off_yds"] = lof.penalty_yards.fillna(10).clip(0, 15).to_numpy(float)
    L["pen_off_yds"] = off.penalty_yards.fillna(5).clip(0, 15).to_numpy(float)
    L["pen_def_yds"] = de.penalty_yards.fillna(5).clip(0, 50).to_numpy(float)
    L["pen_def_first"] = de.first_down_penalty.fillna(0).to_numpy(float)

    # --- clock: seconds until the next snap, by whether the clock stopped and the game situation
    c = _clock_rows(p)
    L["clock"] = _binned(c.elapsed, c.stop * 3 + c.sit, 6)
    L["lg"]["pace"] = c.elapsed[(c.stop == 0) & (c.sit == 0)].mean()
    return L


N_RUSH_BINS = 13
RUN_FAR = {0: 50, 11: 70, 12: 80}  # open-field bins: drawn from runs that started at least this far out
# QB designed runs inside the 20 (sneaks, keepers) score more often than other runners' from the same spot (2021-25:
# 66% vs 56% from the 1, 25% vs 11% from 6-10 yards out), so they get their own bins: the 1, 2, 3-5, 6-10, 11-20
QB_RUN_EDGES = [1, 2, 3, 6, 11]
N_RUN_DIST = N_RUSH_BINS + len(QB_RUN_EDGES)


def qb_rush_bin(yl):
    """Bin for a QB designed run from yard line yl (only used inside the 20)."""
    return N_RUSH_BINS + np.searchsorted(QB_RUN_EDGES, yl, "right") - 1


def rush_bin(yl, togo):
    """Which real runs a designed run's gain is drawn from: inside the 10 by exact distance to the goal line
    (2-7, any down), otherwise short yardage (1) or by field position (8-10: 11-29 yards out, 0: 30-49, 11: 50-69,
    12: 70+). Red-zone bins use runs from the same spot (defenses squeeze them: they score less often than
    open-field gains capped at the goal line would); open-field bins use runs that started farther out (RUN_FAR),
    so the end zone never cut their long gains short."""
    yl, togo = np.asarray(yl), np.asarray(togo)
    gbin = 2 + np.searchsorted(GOAL_YL_EDGES, yl, "right") - 1
    band = np.clip(np.searchsorted(RUN_YL_EDGES, yl, "right") - 1, 0, 5)
    field = np.array([8, 9, 10, 0, 11, 12])[band]
    return np.where(yl <= 10, gbin, np.where(togo <= 2, 1, field))


def yac_bin(air, yl):
    """Yards-after-catch distribution for a catch at depth air from yard line yl: by depth of the catch and by
    the room in front of it (see YAC_ROOM)."""
    a = np.searchsorted(AIR_EDGES, air, "right") - 1
    r = np.searchsorted(YAC_ROOM[1:], np.asarray(yl) - np.asarray(air), "right")
    return a * len(YAC_ROOM) + r


def clock_situation(half, hsr, sd):
    """0 normal tempo, 1 hurry-up (end of first half, or trailing late), 2 leading late and milking the clock."""
    hurry = ((half == 0) & (hsr <= 120)) | ((half == 1) & (sd < 0) & (hsr <= 300)) | \
            ((half == 1) & (sd <= -9) & (hsr <= 600))
    milk = (half == 1) & (sd > 0) & (hsr <= 480) & ~hurry
    return np.where(hurry, 1, np.where(milk, 2, 0))


# =====================================================================================================
# Team and player tendencies (rolling, strictly before each game)
# =====================================================================================================
def _roll(df, key, cols, window):
    df = df.sort_values([key, "game_date"])
    r = df.groupby(key)[cols].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    return pd.concat([df[[key, "game_date"]], r], axis=1)


def _near_goal(plays, who, inside):
    """Per player-game: touches inside the opponent's `inside` yard line (near) and the number his share of the
    team's touches that game would give (exp), rolled over his last 17 games."""
    x = plays.assign(near=(plays.yardline_100 <= inside).astype(float))
    pg = x.groupby([who, "posteam", "game_id", "game_date"]).agg(n=("near", "size"), near=("near", "sum")).reset_index()
    tm = x.groupby(["posteam", "game_id"]).agg(t_n=("near", "size"), t_near=("near", "sum"))
    pg = pg.join(tm, on=["posteam", "game_id"])
    pg["exp"] = pg.n / pg.t_n * pg.t_near
    return _roll(pg.rename(columns={who: "player_id"}), "player_id", ["near", "exp"], 17)


def tendency_tables(pbp, L):
    """Per-game sums rolled over recent games for QBs, receivers, rushers, offenses and defenses."""
    p = pbp.copy()
    p["game_date"] = pd.to_datetime(p.game_date)
    snaps = p[p.play_type.isin(["pass", "run"]) & p.down.notna()]
    passes = snaps[snaps.pass_attempt == 1].passer_player_id.value_counts()
    qb_ids = set(passes[passes >= 20].index)
    db = snaps[snaps.qb_dropback == 1].copy()
    db["qb"] = db.passer_player_id.fillna(db.rusher_player_id)
    att = db[(db.pass_attempt == 1) & (db.sack != 1)]
    T = {}

    q = db.assign(scr_yds=np.where(db.qb_scramble == 1, db.yards_gained, 0.0),
                  att=((db.pass_attempt == 1) & (db.sack != 1)).astype(float),
                  cp_n=(((db.pass_attempt == 1) & (db.sack != 1)) & db.cp.notna()).astype(float),
                  cp_s=np.where((db.pass_attempt == 1) & (db.sack != 1) & db.cp.notna(), db.cp, 0.0),
                  cmp_cp=np.where((db.pass_attempt == 1) & (db.sack != 1) & db.cp.notna(), db.complete_pass, 0.0))
    qg = q.groupby(["qb", "game_id", "game_date"]).agg(
        db=("qb_dropback", "size"), sacks=("sack", "sum"), scr=("qb_scramble", "sum"), scr_yds=("scr_yds", "sum"),
        att=("att", "sum"), ints=("interception", "sum"), cp_n=("cp_n", "sum"), cp_s=("cp_s", "sum"),
        cmp_cp=("cmp_cp", "sum")).reset_index()
    dr = snaps[(snaps.play_type == "run") & (snaps.qb_scramble != 1) & snaps.rusher_player_id.isin(qb_ids)]
    dg = dr.groupby(["rusher_player_id", "game_id"]).agg(qruns=("yards_gained", "size"), qryds=("yards_gained", "sum"))
    qg = qg.join(dg, on=["qb", "game_id"]).fillna({"qruns": 0, "qryds": 0})
    qg["games"] = 1.0
    T["qb"] = _roll(qg.rename(columns={"qb": "player_id"}), "player_id",
                    ["db", "sacks", "scr", "scr_yds", "att", "ints", "cp_n", "cp_s", "cmp_cp", "qruns", "qryds",
                     "games"], 17)

    r = att[att.receiver_player_id.notna()].copy()
    r["air_n"] = r.air_yards.notna().astype(float)
    r["air_s"] = r.air_yards.fillna(0)
    r["cp_n"] = r.cp.notna().astype(float)
    r["cp_s"] = r.cp.fillna(0)
    r["cmp_cp"] = np.where(r.cp.notna(), r.complete_pass, 0.0)
    ok = (r.complete_pass == 1) & r.yards_after_catch.notna() & r.xyac_mean_yardage.notna()
    r["yac_s"] = np.where(ok, r.yards_after_catch, 0.0)
    r["xyac_s"] = np.where(ok, r.xyac_mean_yardage, 0.0)
    rg = r.groupby(["receiver_player_id", "game_id", "game_date"])[
        ["air_n", "air_s", "cp_n", "cp_s", "cmp_cp", "yac_s", "xyac_s"]].sum().reset_index()
    T["rcv"] = _roll(rg.rename(columns={"receiver_player_id": "player_id"}), "player_id",
                     ["air_n", "air_s", "cp_n", "cp_s", "cmp_cp", "yac_s", "xyac_s"], 17)

    ru = snaps[(snaps.play_type == "run") & (snaps.qb_scramble != 1) & snaps.rusher_player_id.notna()]
    ug = ru.groupby(["rusher_player_id", "game_id", "game_date"]).agg(car=("yards_gained", "size"),
                                                                      yds=("yards_gained", "sum")).reset_index()
    T["rsh"] = _roll(ug.rename(columns={"rusher_player_id": "player_id"}), "player_id", ["car", "yds"], 17)

    # near the goal line some players get the ball more than their usual share (a short-yardage back, a QB who
    # sneaks, a big target): touches there vs. what the player's share of all carries/targets would have given him
    T["rsh_gl"] = _near_goal(ru, "rusher_player_id", GL_RUN)
    T["rcv_rz"] = _near_goal(r, "receiver_player_id", RZ_PASS)

    # offense: pass rate over expected, tempo, 4th-down aggressiveness. nflverse's expected pass rate runs 1-2%
    # high league-wide in recent seasons, so measure teams against the league (previous season's average, so
    # nothing from later in the season leaks in)
    o = snaps[snaps.xpass.notna()]
    lg_poe = (o.qb_dropback - o.xpass).groupby(o.season).mean()
    prev = lg_poe.shift(1).fillna(lg_poe)
    o = o.assign(poe=o.qb_dropback - o.xpass - o.season.map(prev))
    og = o.groupby(["posteam", "game_id", "game_date"]).agg(poe_s=("poe", "sum"), poe_n=("poe", "size")).reset_index()
    c = _clock_rows(p)
    pace = c[(c.stop == 0) & (c.sit == 0)].groupby(["posteam", "game_id"]).elapsed.agg(pace_s="sum", pace_n="size")
    og = og.join(pace, on=["posteam", "game_id"]).fillna({"pace_s": 0, "pace_n": 0})
    f4 = p[(p.down == 4) & p.play_type.isin(["pass", "run", "punt", "field_goal"])].copy()
    pr = L["fourth_model"].predict(f4[L["fourth_cols"]].to_numpy(float))
    f4["go_x"] = pr[:, 0]
    f4["go"] = f4.play_type.isin(["pass", "run"]).astype(float)
    fg4 = f4.groupby(["posteam", "game_id"]).agg(go_s=("go", "sum"), gox_s=("go_x", "sum"), go_n=("go", "size"))
    og = og.join(fg4, on=["posteam", "game_id"]).fillna({"go_s": 0, "gox_s": 0, "go_n": 0})
    T["off"] = _roll(og.rename(columns={"posteam": "team"}), "team",
                     ["poe_s", "poe_n", "pace_s", "pace_n", "go_s", "gox_s", "go_n"], 10)

    # defense: pass rush, coverage (completions over expected, yards per catch), interceptions, run defense
    dd = db.assign(att=((db.pass_attempt == 1) & (db.sack != 1)).astype(float),
                   cp_n=((db.pass_attempt == 1) & (db.sack != 1) & db.cp.notna()).astype(float),
                   cp_s=np.where((db.pass_attempt == 1) & (db.sack != 1) & db.cp.notna(), db.cp, 0.0),
                   cmp_cp=np.where((db.pass_attempt == 1) & (db.sack != 1) & db.cp.notna(), db.complete_pass, 0.0),
                   cmp_yds=np.where(db.complete_pass == 1, db.yards_gained, 0.0))
    dg = dd.groupby(["defteam", "game_id", "game_date"]).agg(
        db=("qb_dropback", "size"), sacks=("sack", "sum"), att=("att", "sum"), ints=("interception", "sum"),
        cp_n=("cp_n", "sum"), cp_s=("cp_s", "sum"), cmp_cp=("cmp_cp", "sum"), cmp=("complete_pass", "sum"),
        cmp_yds=("cmp_yds", "sum")).reset_index()
    rd = ru.groupby(["defteam", "game_id"]).agg(car=("yards_gained", "size"), ryds=("yards_gained", "sum"))
    dg = dg.join(rd, on=["defteam", "game_id"]).fillna({"car": 0, "ryds": 0})
    T["dfn"] = _roll(dg.rename(columns={"defteam": "team"}), "team",
                     ["db", "sacks", "att", "ints", "cp_n", "cp_s", "cmp_cp", "cmp", "cmp_yds", "car", "ryds"], 12)
    return T


def asof(table, key, keys, dates):
    """Rolling sums for each (key, date) as of the last game strictly before that date (NaN if none)."""
    q = pd.DataFrame({key: list(keys), "date": pd.to_datetime(list(dates))}).reset_index()
    t = table.sort_values("game_date")
    m = pd.merge_asof(q.sort_values("date"), t, left_on="date", right_on="game_date", by=key,
                      allow_exact_matches=False)
    return m.set_index("index").sort_index()


# =====================================================================================================
# One game's parameters
# =====================================================================================================
POS_ADOT = {"WR": 10.5, "TE": 7.0, "RB": 0.8, "FB": 1.5, "QB": 5.0}
POS_TGT = {"WR": 0.06, "TE": 0.04, "RB": 0.03, "FB": 0.01, "QB": 0.0}
POS_CAR = {"RB": 0.08, "FB": 0.01, "WR": 0.005, "TE": 0.0, "QB": 0.0}
# Near the goal line: designed runs inside the 10 and targets inside the 20 go to each player in proportion to
# his usual share times (touches there + K x m) / (touches his share predicts + K), where m is what players at his
# position get there league-wide relative to their share (2021-24: QBs 1.67x the designed runs, wide receivers
# 0.54x; tight ends 1.13x the targets). K shrinks small samples toward m.
GL_RUN, RZ_PASS = 10, 20
K_GL, K_RZ = 5.0, 8.0
K_GL_QB = 0.5  # QBs: own goal-line run rate, barely shrunk (2025-26 sim backtest: QB TD log loss 0.4004 -> 0.3979)
GL_POS = {"QB": 1.67, "RB": 0.95, "FB": 0.95, "WR": 0.54, "TE": 0.54}
RZ_POS = {"RB": 0.96, "FB": 0.96, "WR": 0.96, "TE": 1.13}


def game_params(players, home, away, date, T, L):
    """players: rows for this game (both teams) with player_id, player_display_name, team, position,
    target_share_norm, carry_share_norm, is_starting_qb. Returns the dict the engine reads."""
    lg = L["lg"]
    pl = players.reset_index(drop=True).copy()
    teams = [away, home]
    pl["t"] = pl.team.map({away: 0, home: 1})
    pl = pl[pl.t.notna()].reset_index(drop=True)
    pl["t"] = pl.t.astype(int)
    G = {"names": pl.player_display_name.tolist(), "pid": pl.player_id.tolist(), "team": pl.t.to_numpy(),
         "pos": pl.position.tolist(), "n": len(pl)}

    rc = asof(T["rcv"], "player_id", pl.player_id, [date] * len(pl)).fillna(0)
    rs = asof(T["rsh"], "player_id", pl.player_id, [date] * len(pl)).fillna(0)
    pos_adot = pl.position.map(POS_ADOT).fillna(5.0).to_numpy()
    adot = (rc.air_s.to_numpy() + 15 * pos_adot) / (rc.air_n.to_numpy() + 15)
    croe = (rc.cmp_cp - rc.cp_s - lg["cpoe"] * rc.cp_n).to_numpy() / (rc.cp_n.to_numpy() + 50)
    croe_logit = _logit(np.clip(lg["cmp"] + croe, 0.05, 0.95)) - _logit(lg["cmp"])
    r = lg["yac_ratio"]
    yac = (rc.yac_s.to_numpy() + 25 * 5.0 * r) / (rc.xyac_s.to_numpy() + 25 * 5.0) / r
    ypc = (rs.yds.to_numpy() + 80 * lg["ypc"]) / (rs.car.to_numpy() + 80) / lg["ypc"]
    gl = asof(T["rsh_gl"], "player_id", pl.player_id, [date] * len(pl)).fillna(0)
    rz = asof(T["rcv_rz"], "player_id", pl.player_id, [date] * len(pl)).fillna(0)
    k_gl = np.where(pl.position == "QB", K_GL_QB, K_GL)
    gl_m = ((gl.near + k_gl * pl.position.map(GL_POS).fillna(1.0)) / (gl.exp + k_gl)).to_numpy()
    rz_m = ((rz.near + K_RZ * pl.position.map(RZ_POS).fillna(1.0)) / (rz.exp + K_RZ)).to_numpy()

    G["qb"] = np.full(2, -1)
    qbp = {}
    for t in (0, 1):
        q = pl[(pl.t == t) & (pl.is_starting_qb == 1)]
        qid = q.player_id.iloc[0] if len(q) else None
        G["qb"][t] = q.index[0] if len(q) else -1
        h = asof(T["qb"], "player_id", [qid], [date]).iloc[0] if qid is not None else None
        g = (lambda c: 0.0 if h is None or pd.isna(h[c]) else float(h[c]))
        # QBs with little history get a slightly-below-average prior (they are usually backups or rookies)
        sack = (g("sacks") + 150 * lg["sack"] * 1.12) / (g("db") + 150)
        scr = (g("scr") + 150 * lg["scramble"]) / (g("db") + 150)
        scr_m = (g("scr_yds") + 30 * lg["scr_ypc"]) / (g("scr") + 30) / lg["scr_ypc"]
        cpoe = (g("cmp_cp") - g("cp_s") - lg["cpoe"] * g("cp_n") - 200 * 0.02) / (g("cp_n") + 200)
        ints = (g("ints") + 300 * lg["int"] * 1.15) / (g("att") + 300) / lg["int"]
        runs_pg = (g("qruns") + 4 * 1.2) / (g("games") + 4)
        qypc = (g("qryds") + 40 * lg["ypc"]) / (g("qruns") + 40) / lg["ypc"]
        qbp[t] = dict(sack=sack, scr=scr, scr_m=scr_m, cpoe=_logit(np.clip(lg["cmp"] + cpoe, .05, .95)) - _logit(lg["cmp"]),
                      int=ints, run_share=float(np.clip(runs_pg / 24.0, 0.0, 0.4)), qypc=qypc)

    off = asof(T["off"], "team", teams, [date, date])
    dfn = asof(T["dfn"], "team", teams, [date, date])
    G["proe"], G["pace"], G["go"] = np.zeros(2), np.ones(2), np.zeros(2)
    G["sack_p"], G["scr_p"], G["scr_m"], G["cpoe"], G["int_m"] = (np.zeros(2) for _ in range(5))
    G["def_cp"], G["def_pass"], G["def_rush"] = np.zeros(2), np.ones(2), np.ones(2)
    for t in (0, 1):
        o, dfd = off.iloc[t].fillna(0), dfn.iloc[1 - t].fillna(0)  # offense t faces defense 1-t
        G["proe"][t] = (o.poe_s / (o.poe_n + 250)) / (lg["pass_rate"] * (1 - lg["pass_rate"]))
        G["pace"][t] = (o.pace_s + 150 * lg["pace"]) / (o.pace_n + 150) / lg["pace"]
        G["go"][t] = (_logit(np.clip((o.go_s + 30 * 0.3) / (o.go_n + 30), .01, .99))
                      - _logit(np.clip((o.gox_s + 30 * 0.3) / (o.go_n + 30), .01, .99)))
        ds = (dfd.sacks + 200 * lg["sack"]) / (dfd.db + 200)
        po, pd_, pl_ = qbp[t]["sack"], ds, lg["sack"]
        G["sack_p"][t] = (po * pd_ / pl_) / (po * pd_ / pl_ + (1 - po) * (1 - pd_) / (1 - pl_))  # log5
        G["scr_p"][t], G["scr_m"][t] = qbp[t]["scr"], qbp[t]["scr_m"]
        dcp = (dfd.cmp_cp - dfd.cp_s - lg["cpoe"] * dfd.cp_n) / (dfd.cp_n + 300)
        G["def_cp"][t] = _logit(np.clip(lg["cmp"] + dcp, .05, .95)) - _logit(lg["cmp"])
        G["cpoe"][t] = qbp[t]["cpoe"]
        G["int_m"][t] = qbp[t]["int"] * (dfd.ints + 400 * lg["int"]) / (dfd.att + 400) / lg["int"]
        G["def_pass"][t] = (dfd.cmp_yds + 150 * lg["ypc_pass"]) / (dfd.cmp + 150) / lg["ypc_pass"]
        G["def_rush"][t] = (dfd.ryds + 200 * lg["ypc"]) / (dfd.car + 200) / lg["ypc"]

    # usage among the players active in this game
    tgt = pl.target_share_norm.to_numpy(float)
    tgt = np.where(np.isnan(tgt), pl.position.map(POS_TGT).fillna(0.0).to_numpy(), tgt)
    car = pl.carry_share_norm.to_numpy(float)
    car = np.where(np.isnan(car), pl.position.map(POS_CAR).fillna(0.0).to_numpy(), car)
    is_qb = (pl.position == "QB").to_numpy()
    tgt[is_qb] = 0.0
    car[is_qb] = 0.0
    R = max(int((pl.t == t).sum()) for t in (0, 1))
    G["rcv_cum"], G["rcv_idx"] = np.ones((2, R)), np.zeros((2, R), int)
    G["adot"], G["adot_bin"], G["croe"], G["yac"] = np.zeros((2, R)), np.zeros((2, R), int), np.zeros((2, R)), np.ones((2, R))
    G["rsh_cum"], G["rsh_idx"], G["ypc"] = np.ones((2, R + 1)), np.zeros((2, R + 1), int), np.ones((2, R + 1))
    G["rcv_rz_m"], G["rsh_gl_m"] = np.ones((2, R)), np.ones((2, R + 1))
    for t in (0, 1):
        ix = np.flatnonzero(pl.t.to_numpy() == t)
        w = tgt[ix] / tgt[ix].sum() if tgt[ix].sum() > 0 else np.ones(len(ix)) / len(ix)
        n = len(ix)
        G["rcv_cum"][t, :n] = np.cumsum(w)
        G["rcv_cum"][t, n - 1:] = 1.0 + 1e-9
        G["rcv_idx"][t, :n] = ix
        G["adot"][t, :n] = adot[ix]
        G["adot_bin"][t, :n] = np.searchsorted(ADOT_EDGES, adot[ix], "right") - 1
        G["croe"][t, :n], G["yac"][t, :n] = croe_logit[ix], yac[ix]
        G["rcv_rz_m"][t, :n] = rz_m[ix]
        # designed runs: the QB's share, the rest split by carry share
        qs = qbp[t]["run_share"] if G["qb"][t] >= 0 else 0.0
        c = car[ix] / car[ix].sum() * (1 - qs) if car[ix].sum() > 0 else np.ones(n) / n * (1 - qs)
        shares, who, eff = list(c), list(ix), list(ypc[ix])
        if G["qb"][t] >= 0:
            shares.append(qs)
            who.append(G["qb"][t])
            eff.append(qbp[t]["qypc"])
        m = len(shares)
        G["rsh_cum"][t, :m] = np.cumsum(shares)
        G["rsh_cum"][t, m - 1:] = 1.0 + 1e-9
        G["rsh_idx"][t, :m], G["ypc"][t, :m] = who, eff
        G["rsh_gl_m"][t, :m] = gl_m[who]
    G["rcv_w"] = np.diff(np.minimum(G["rcv_cum"], 1.0), axis=1, prepend=0.0)
    G["rsh_w"] = np.diff(np.minimum(G["rsh_cum"], 1.0), axis=1, prepend=0.0)
    G["p_exit"] = pl.position.map(P_EXIT).fillna(0.05).to_numpy()
    return G


# Measured from 2022-25 player-games: target shares vary 1.5x and carry shares 2.8x more from game to game than
# fixed shares would (Dirichlet concentrations below), and regulars play under half their usual snaps in ~6% of
# games (injuries, benchings, blowouts), QBs most often.
A_TGT, A_CAR = 56.0, 13.0
P_EXIT = {"QB": 0.07, "RB": 0.04, "FB": 0.04, "WR": 0.05, "TE": 0.045}
BACKUP_QB = dict(cp=-0.3, int=1.3, sack=1.2)  # when the starter has left: a below-average replacement


# =====================================================================================================
# The engine: all simulations of one game advance together, one snap at a time
# =====================================================================================================
GAME_SHOCK = 0.12  # sd of each offense's game-day efficiency (log scale); sized to real score variance


class Sim:
    def __init__(self, G, L, n, k_eff, rng, shock=GAME_SHOCK):
        self.G, self.L, self.n, self.rng = G, L, n, rng
        self.k = np.asarray(k_eff, dtype=float)
        # good days and bad days: one efficiency draw per offense per simulated game, shared by all its plays
        self.day = np.exp(rng.normal(0.0, shock, (n, 2)) - shock ** 2 / 2)
        self.stats = np.zeros((n, G["n"], NSTAT), dtype=np.float32)
        self.score = np.zeros((n, 2), dtype=np.int32)
        self.t = np.full(n, 3600.0)
        self.second = np.zeros(n, bool)
        self.ot = np.zeros(n, bool)
        self.ot_done = np.zeros((n, 2), np.int8)  # overtime possessions each team has finished
        self.opening_kicker = rng.integers(0, 2, n)
        self.pos = self.opening_kicker.copy()  # during a kickoff, pos = kicking team
        self.phase = np.full(n, KO, np.int8)
        self.down = np.ones(n, np.int16)
        self.togo = np.full(n, 10, np.int16)
        self.yl = np.full(n, 75, np.int16)
        self.done = np.zeros(n, bool)
        self.plays = np.zeros((n, 2), np.int16)
        self.ev = {k: np.zeros((n, 2), np.int16) for k in
                   ("td", "fg_att", "fg_made", "punt", "int", "fum", "sack", "downs", "go4", "drives",
                    "d3_att", "d3_conv", "rz_trips", "rz_td", "dst_td")}
        self.in_rz = np.zeros(n, bool)
        # yardage multipliers per simulated game and offense: market calibration x opposing defense x game day
        self.m_pass = self.day * (self.k * G["def_pass"])[None, :]
        self.m_rush = self.day * (self.k * G["def_rush"])[None, :]
        self.air = np.zeros((n, 2))
        # this game's usage: each player's share redrawn around his expected share
        self.rcv_w = self._dirichlet(G["rcv_w"], A_TGT, n)
        self.rsh_w = self._dirichlet(G["rsh_w"], A_CAR, n)
        # ...tilted near the goal line toward the players who usually get the ball there
        self.rcv_w_rz = self._tilt(self.rcv_w, G["rcv_rz_m"])
        self.rsh_w_gl = self._tilt(self.rsh_w, G["rsh_gl_m"])
        self.first_td = np.full(n, -1)  # player who scored the game's first touchdown (-1: nobody tracked)
        # early exits: when (seconds into the game) each player leaves, inf if he plays it out
        leave = rng.random((n, G["n"])) < G["p_exit"][None, :]
        self.exit_at = np.where(leave, rng.uniform(0, 3600, (n, G["n"])), np.inf)

    def _dirichlet(self, w, alpha, n):
        g = self.rng.gamma(np.maximum(alpha * w, 1e-6)[None], size=(n,) + w.shape) * (w > 0)[None]
        return g / np.maximum(g.sum(-1, keepdims=True), 1e-12)

    @staticmethod
    def _tilt(w, m):
        x = w * m[None]
        return x / np.maximum(x.sum(-1, keepdims=True), 1e-12)

    def clock(self, I):
        """Seconds played so far (regulation, then overtime)."""
        return np.where(self.ot[I], 4200 - self.t[I], 3600 - self.t[I])

    def pick(self, w, idx, I):
        """Sample one player per row from per-row shares w (len(I), R), skipping players who have left."""
        on = self.exit_at[I[:, None], idx] > self.clock(I)[:, None]
        w_on = w * on
        s = w_on.sum(1, keepdims=True)
        w = np.where(s > 0, w_on / np.maximum(s, 1e-12), w)  # everyone gone (rare): ignore exits
        c = np.cumsum(w, 1)
        c[:, -1] = 1.0 + 1e-9
        return (self.rng.random(len(I))[:, None] >= c).sum(1)

    # ---- helpers
    def hsr(self, I):
        return np.where(self.second[I] | self.ot[I], self.t[I], self.t[I] - 1800)

    def half(self, I):
        return (self.second[I] | self.ot[I]).astype(int)

    def end_possession(self, I, team):
        o = self.ot[I]
        if o.any():
            self.ot_done[I[o], np.asarray(team)[o]] += 1

    def gain_possession(self, I, team, yl):
        """team takes over at yl after the other side's possession ended (turnover, punt, downs, missed kick)."""
        if self.phase[I].size and (self.phase[I] == SCRIM).any():
            S = self.phase[I] == SCRIM
            self.end_possession(I[S], 1 - np.asarray(team)[S])
        self.ev["drives"][I, team] += 1
        self.in_rz[I] = False
        self.pos[I], self.yl[I] = team, np.clip(yl, 1, 99)
        self.down[I], self.togo[I] = 1, np.minimum(10, self.yl[I])
        self.phase[I] = SCRIM

    def touchdown(self, I, team):
        self.end_possession(I, team)
        self.ev["td"][I, team] += 1
        rz = self.in_rz[I] & (self.pos[I] == team)
        self.ev["rz_td"][I[rz], np.asarray(team)[rz]] += 1
        self.score[I, team] += 6
        self.pos[I] = team
        self.phase[I] = PAT

    def run_clock(self, I, stopped, sit, offense):
        el = _draw(self.L["clock"], stopped.astype(int) * 3 + sit, self.rng)
        el = np.where((~stopped) & (sit == 0), el * self.G["pace"][offense], el)
        self.t[I] -= el

    # ---- period ends (checked before every snap or kickoff)
    def check_periods(self):
        I = np.flatnonzero(~self.done & (self.phase != PAT))
        if not I.size:
            return
        reg = ~self.ot[I]
        h1 = I[reg & ~self.second[I] & (self.t[I] <= 1800)]
        if h1.size:  # halftime: the team that kicked first receives
            self.second[h1], self.t[h1], self.phase[h1] = True, 1800.0, KO
            self.pos[h1] = 1 - self.opening_kicker[h1]
        end = I[reg & self.second[I] & (self.t[I] <= 0)]
        if end.size:
            tied = self.score[end, 0] == self.score[end, 1]
            self.done[end[~tied]] = True
            o = end[tied]
            self.ot[o], self.t[o], self.phase[o] = True, 600.0, KO
            self.pos[o] = self.rng.integers(0, 2, o.size)
        J = I[self.ot[I] & ~self.done[I]]
        if J.size:  # overtime: both teams get a possession, then the next score wins; 10 minutes or a tie
            over = (self.t[J] <= 0) | ((self.ot_done[J].min(1) >= 1) & (self.score[J, 0] != self.score[J, 1]))
            self.done[J[over]] = True

    # ---- kickoffs (incl. onside kicks when trailing late)
    def kickoff(self, I):
        rng, k = self.rng, self.pos[I]
        r = 1 - k
        sd = self.score[I, k] - self.score[I, r]
        late = (self.second[I] | self.ot[I]) & (self.t[I] <= 300) & ~self.ot[I]
        onside = late & (sd < 0) & (sd >= -16) & (rng.random(I.size) < np.where(self.t[I] <= 150, 0.9, 0.4))
        got = onside & (rng.random(I.size) < 0.10)
        start = _draw1(self.L["kickoff"], I.size, rng)
        ret_td = (start <= 0) & ~onside
        self.t[I] -= np.where(start >= 65, 0.0, 5.0)
        if ret_td.any():
            self.ev["dst_td"][I[ret_td], r[ret_td]] += 1
            self.touchdown(I[ret_td], r[ret_td])
        rest = ~ret_td
        team = np.where(got, k, r)
        yl = np.where(onside, np.where(got, 55, 47), start)
        self.gain_possession(I[rest], team[rest], yl[rest])

    def pat(self, I):
        rng, o = self.rng, self.pos[I]
        sd = self.score[I, o] - self.score[I, 1 - o]
        late = (self.second[I] & (self.t[I] <= 900)) | self.ot[I]
        go2 = rng.random(I.size) < self.L["go2"][late.astype(int), np.clip(sd, -20, 20) + 20]
        ok = rng.random(I.size) < np.where(go2, self.L["two_rate"], self.L["xp"])
        self.score[I, o] += np.where(ok, np.where(go2, 2, 1), 0)
        self.phase[I] = KO  # the scoring team kicks off

    # ---- a snap from scrimmage
    def scrimmage(self, I):
        L, G, rng = self.L, self.G, self.rng
        o = self.pos[I]
        d = 1 - o
        sd = self.score[I, o] - self.score[I, d]
        half, hsr = self.half(I), self.hsr(I)
        down, togo, yl = self.down[I], self.togo[I], self.yl[I]

        # kneel to finish a won game or run out the first half
        kneel = ((half == 1) & ~self.ot[I] & (sd > 0) & (down <= 3) & (self.t[I] <= 40 * (4 - down))) | \
                ((half == 0) & (hsr <= 25) & (yl >= 60))
        K = I[kneel]
        if K.size:
            qb = G["qb"][self.pos[K]]
            has = qb >= 0
            self.stats[K[has], qb[has], 3] -= 1
            self.stats[K[has], qb[has], 4] += 1
            self.yl[K] += 1
            self.togo[K] += 1
            self.down[K] += 1
            self.t[K] -= 40.0
            self.turnover_on_downs(K[self.down[K] > 4])

        # 4th down: go, punt or kick. Also kick on any down when the half is about to end in field-goal range
        # (and in the 4th quarter only if 3 points ties or wins)
        idx = (TOGO_MAP[np.clip(togo, 0, 99)], YL_MAP[np.clip(yl, 0, 99)], SD_MAP[np.clip(sd, -60, 60) + 60], half,
               HSR_MAP[np.clip(hsr, 0, 1800).astype(int)])
        choice = np.zeros(I.size, int)
        clock_fg = (~kneel) & (hsr <= 10) & (yl <= 37) & ((half == 0) | ((sd <= 0) & (sd >= -3)))
        choice[clock_fg] = 2
        f4 = (down == 4) & ~kneel & ~clock_fg
        if f4.any():
            lg4 = L["fourth"][tuple(a[f4] for a in idx)].copy()
            lg4[:, 0] += G["go"][o[f4]]
            pr = np.exp(lg4 - lg4.max(1, keepdims=True))
            pr /= pr.sum(1, keepdims=True)
            choice[f4] = (rng.random(f4.sum())[:, None] > np.cumsum(pr, 1)).sum(1)
        self.punt(I[f4 & (choice == 1)])
        self.field_goal(I[(f4 | clock_fg) & (choice == 2)])
        go = I[f4 & (choice == 0)]
        self.ev["go4"][go, self.pos[go]] += 1
        play = ~kneel & (choice == 0)
        if not play.any():
            return
        I, o, d, sd, half, hsr = I[play], o[play], d[play], sd[play], half[play], hsr[play]
        down, togo, yl = self.down[I], self.togo[I], self.yl[I]
        idx = tuple(a[play] for a in idx)
        self.plays[I, o] += 1
        new_rz = (yl <= 20) & ~self.in_rz[I]
        self.ev["rz_trips"][I[new_rz], o[new_rz]] += 1
        self.in_rz[I[new_rz]] = True

        # penalties that replace the play
        u = rng.random(I.size)
        off_pen = u < L["pen_off"]
        def_pen = (~off_pen) & (u < L["pen_off"] + L["pen_def"])
        if off_pen.any():
            P = I[off_pen]
            y = np.minimum(_draw1(L["pen_off_yds"], P.size, rng), 99 - self.yl[P]).astype(np.int16)
            self.yl[P] += y
            self.togo[P] += y
            self.t[P] -= 5.0
        if def_pen.any():
            P = I[def_pen]
            j = rng.integers(0, len(L["pen_def_yds"]), P.size)
            y = np.minimum(L["pen_def_yds"][j], np.maximum(self.yl[P] // 2, 1)).astype(np.int16)
            first = (L["pen_def_first"][j] == 1) | (y >= self.togo[P])
            self.yl[P] -= y
            self.togo[P] = np.where(first, np.minimum(10, self.yl[P]), self.togo[P] - y)
            self.down[P] = np.where(first, 1, self.down[P])
            self.t[P] -= 5.0
        live = ~(off_pen | def_pen)
        I, o, d, sd, half, hsr = I[live], o[live], d[live], sd[live], half[live], hsr[live]
        down, togo, yl = down[live], togo[live], yl[live]
        idx = (np.clip(down - 1, 0, 3),) + tuple(a[live] for a in idx)

        # pass or run: league tendency for the situation + this coach's pass rate over expected
        is_pass = rng.random(I.size) < _sigmoid(L["xpass"][idx] + G["proe"][o])
        gain = np.zeros(I.size)
        stopped = np.zeros(I.size, bool)
        lost = np.zeros(I.size, bool)          # fumble lost or interception
        new_yl = np.zeros(I.size)              # possession-change spot for turnovers (other team's view)
        was_sack = np.zeros(I.size, bool)
        was_int = np.zeros(I.size, bool)
        scorer = np.full(I.size, -1)           # who scores if the snap ends in the end zone (-1: a backup QB)
        caught = np.zeros(I.size, bool)        # by a catch (receiving TD) rather than a run
        sit = clock_situation(half, hsr, sd)

        J = np.flatnonzero(is_pass)
        if J.size:
            oj, dj, ylj = o[J], d[J], yl[J]
            qb = G["qb"][oj]
            # is the starter still in? if not, a backup throws (his stats aren't anyone's props)
            starter = (qb >= 0) & (self.exit_at[I[J], np.maximum(qb, 0)] > self.clock(I[J]))
            qb = np.where(starter, qb, -1)
            sack_p = np.where(starter, G["sack_p"][oj], np.minimum(G["sack_p"][oj] * BACKUP_QB["sack"], 0.3))
            u = rng.random(J.size)
            sack = u < sack_p
            scr = (~sack) & (u < sack_p + G["scr_p"][oj])
            att = ~(sack | scr)
            was_sack[J[sack]] = True
            if sack.any():
                g = _draw1(L["sack"], sack.sum(), rng)
                gain[J[sack]] = g
                lost[J[sack]] = rng.random(sack.sum()) < L["fum_sack"]
                new_yl[J[sack]] = 100 - (ylj[sack] - g)  # fumble recovered where the QB went down
            if scr.any():
                g = _draw1(L["scramble"], scr.sum(), rng)
                g = np.where(g > 0, g * G["scr_m"][oj[scr]] * self.k[oj[scr]] * self.day[I[J[scr]], oj[scr]], g)
                g = np.minimum(np.round(g), ylj[scr])
                gain[J[scr]] = g
                rows, who = I[J[scr]], qb[scr]
                ok = who >= 0
                self.stats[rows[ok], who[ok], 3] += g[ok]
                self.stats[rows[ok], who[ok], 4] += 1
                scorer[J[scr]] = who
                lost[J[scr]] = (g < ylj[scr]) & (rng.random(scr.sum()) < L["fum_run"])  # a TD can't be fumbled away
                new_yl[J[scr]] = 100 - (ylj[scr] - g)
            if att.any():
                A = J[att]
                oa, da, yla = o[A], d[A], yl[A]
                w = np.where((yla <= RZ_PASS)[:, None], self.rcv_w_rz[I[A], oa], self.rcv_w[I[A], oa])
                ri = self.pick(w, G["rcv_idx"][oa], I[A])
                rec = G["rcv_idx"][oa, ri]
                air = np.round(G["adot"][oa, ri] + _draw(L["air_resid"], G["adot_bin"][oa, ri], rng))
                rzp = yla <= 25  # red zone: routes compress, use real throws from that field position
                if rzp.any():
                    cell = (np.searchsorted(RZ_YL_EDGES, yla[rzp], "right") - 1) * 3 + \
                           (np.searchsorted(RZ_TYPE_EDGES, G["adot"][oa[rzp], ri[rzp]], "right") - 1)
                    air[rzp] = _draw(L["air_rz"], cell, rng)
                air = np.clip(air, -10, yla + 9).astype(int)
                ai, yi = np.clip(air + 15, 0, 85), np.clip(yla - 1, 0, 98)
                st = starter[att]
                pc = _sigmoid(L["cp"][ai, yi] + G["croe"][oa, ri] + np.where(st, G["cpoe"][oa], BACKUP_QB["cp"])
                              + G["def_cp"][oa])
                pi = _sigmoid(L["int"][ai, yi] + np.log(G["int_m"][oa] * np.where(st, 1.0, BACKUP_QB["int"])))
                pc = np.minimum(pc, 1 - pi)  # both models are P(outcome) over all attempts
                u = rng.random(A.size)
                inter = u < pi
                comp = (~inter) & (u < pi + pc)
                yac = _draw(L["yac"], yac_bin(air, yla), rng)
                yac = np.where(yac > 0, yac * G["yac"][oa, ri], yac)
                g = air + yac
                # team, defense and game-day multipliers stretch yardage; near the goal line they would mostly turn
                # short gains into touchdowns, so there the league's real outcomes stand
                g = np.where((g > 0) & (yla > NO_SCALE_PASS), g * self.m_pass[I[A], oa], g)
                g = np.where(comp, np.minimum(np.round(g), yla), 0.0)
                gain[A] = g
                stopped[A] = ~comp
                rows, qa = I[A], qb[att]
                self.air[rows, oa] += air
                okq = qa >= 0
                self.stats[rows[okq], qa[okq], 1] += 1
                self.stats[rows[okq & comp], qa[okq & comp], 2] += 1
                self.stats[rows[okq & comp], qa[okq & comp], 0] += g[okq & comp]
                self.stats[rows, rec, 7] += 1
                self.stats[rows[comp], rec[comp], 6] += 1
                self.stats[rows[comp], rec[comp], 5] += g[comp]
                scorer[A], caught[A] = np.where(comp, rec, -1), comp
                fum = comp & (g < yla) & (rng.random(A.size) < L["fum_rec"])
                lost[A] = inter | fum
                was_int[A] = inter
                # interception: other team's ball at the catch point minus the return
                ret = _draw1(L["int_return"], A.size, rng)
                spot_int = np.where(air >= yla, 80, 100 - yla + air - ret)  # in the end zone: touchback
                new_yl[A] = np.where(inter, spot_int, 100 - (yla - g))

        R = np.flatnonzero(~is_pass)
        if R.size:
            orr, dr, ylr = o[R], d[R], yl[R]
            w = np.where((ylr <= GL_RUN)[:, None], self.rsh_w_gl[I[R], orr], self.rsh_w[I[R], orr])
            ri = self.pick(w, G["rsh_idx"][orr], I[R])
            who = G["rsh_idx"][orr, ri]
            qb_run = (who == G["qb"][orr]) & (ylr <= 20)
            g = _draw(L["rush"], np.where(qb_run, qb_rush_bin(ylr), rush_bin(ylr, togo[R])), rng)
            g = np.where((g > 0) & (ylr > NO_SCALE_RUN), g * G["ypc"][orr, ri] * self.m_rush[I[R], orr], g)
            g = np.minimum(np.round(g), ylr)
            gain[R] = g
            self.stats[I[R], who, 3] += g
            self.stats[I[R], who, 4] += 1
            scorer[R] = who
            fum = (g < ylr) & (rng.random(R.size) < L["fum_run"])
            lost[R] = fum
            new_yl[R] = 100 - (ylr - g)

        # results of the snap
        self.ev["sack"][I[was_sack], o[was_sack]] += 1
        self.ev["int"][I[was_int], o[was_int]] += 1
        fumbled = lost & ~was_int
        self.ev["fum"][I[fumbled], o[fumbled]] += 1
        after = yl - gain
        td = (~lost) & (after <= 0)
        safety = (~lost) & (after >= 100)
        to = lost & (new_yl > 0)
        pick6 = lost & (new_yl <= 0)
        self.run_clock(I, stopped | td | lost, sit, o)
        if td.any():
            rows, who, rec_td = I[td], scorer[td], caught[td]
            ok = who >= 0
            self.stats[rows[ok], who[ok], np.where(rec_td[ok], TD_REC, TD_RUSH)] += 1
            first = self.ev["td"][rows].sum(1) == 0  # no touchdown yet in this game, by anyone
            self.first_td[rows[first]] = who[first]
            self.touchdown(rows, o[td])
        if pick6.any():
            self.ev["dst_td"][I[pick6], d[pick6]] += 1
            self.touchdown(I[pick6], d[pick6])
        if safety.any():
            S = I[safety]
            self.end_possession(S, o[safety])
            self.score[S, d[safety]] += 2
            self.phase[S] = KO  # the team that gave up the safety kicks
        if to.any():
            self.gain_possession(I[to], d[to], new_yl[to])
        d3 = down == 3
        self.ev["d3_att"][I[d3], o[d3]] += 1
        conv = d3 & ((~lost) & ((gain >= togo) | td))
        self.ev["d3_conv"][I[conv], o[conv]] += 1
        cont = ~(td | safety | lost)
        C = I[cont]
        if C.size:
            g = gain[cont]
            # penalties accepted on top of the play (player stats above stand)
            u = rng.random(C.size)
            ld = u < L["live_def"]
            lo = (~ld) & (u < L["live_def"] + L["live_off"])
            j = rng.integers(0, len(L["live_def_yds"]), C.size)
            extra = np.where(ld, np.minimum(L["live_def_yds"][j], np.maximum(after[cont] - 1, 0)), 0.0)
            extra -= np.where(lo, np.minimum(_draw1(L["live_off_yds"], C.size, rng), np.maximum(99 - after[cont], 0)), 0.0)
            after[cont] -= extra
            g = g + extra
            first = (g >= togo[cont]) | (ld & (L["live_def_first"][j] == 1))
            self.yl[C] = after[cont]
            self.down[C] = np.where(first, 1, down[cont] + 1)
            self.togo[C] = np.where(first, np.minimum(10, after[cont]), togo[cont] - g)
            self.turnover_on_downs(C[self.down[C] > 4])

    def turnover_on_downs(self, I):
        if I.size:
            self.ev["downs"][I, self.pos[I]] += 1
            self.gain_possession(I, 1 - self.pos[I], 100 - self.yl[I])

    def punt(self, I):
        if not I.size:
            return
        o = self.pos[I]
        self.ev["punt"][I, o] += 1
        nyl = _draw(self.L["punt"], np.clip(self.yl[I] // 5, 0, len(PUNT_EDGES) - 1), self.rng)
        self.t[I] -= 9.0
        td = nyl <= 0
        if td.any():
            self.ev["dst_td"][I[td], 1 - o[td]] += 1
            self.touchdown(I[td], 1 - o[td])
        self.gain_possession(I[~td], 1 - o[~td], nyl[~td])

    def field_goal(self, I):
        if not I.size:
            return
        o = self.pos[I]
        dist = np.clip(self.yl[I] + 17, 0, 80)
        made = self.rng.random(I.size) < self.L["fg"][dist]
        self.ev["fg_att"][I, o] += 1
        self.ev["fg_made"][I[made], o[made]] += 1
        self.t[I] -= 5.0
        self.end_possession(I[made], o[made])
        self.score[I[made], o[made]] += 3
        self.phase[I[made]] = KO
        miss = ~made
        self.gain_possession(I[miss], 1 - o[miss], np.minimum(100 - (self.yl[I[miss]] + 7), 80))

    def run(self):
        for _ in range(600):
            self.check_periods()
            if self.done.all():
                break
            for ph, fn in ((KO, self.kickoff), (PAT, self.pat), (SCRIM, self.scrimmage)):
                I = np.flatnonzero(~self.done & (self.phase == ph))
                if I.size:
                    fn(I)
        return self


# Real teams average ~1.0 point a game from defensive/special-teams touchdowns and safeties (2021-25); the
# simulator produces ~0.4 (pick-sixes, return TDs). The market's totals include all of it, so tune the offense
# to the implied total minus the part the simulator doesn't produce, rather than inflating offensive yards.
UNMODELED_PTS = 0.6


def simulate_game(G, L, n=N_SIMS, implied=None, seed=0, calib_n=300, rounds=3):
    """Run the game n times. If implied (away, home) points are given, first tune each offense's efficiency so
    the average simulated score matches them. Returns the finished Sim and the efficiency multipliers."""
    rng = np.random.default_rng(seed)
    k = np.ones(2)
    if implied is not None and np.all(np.isfinite(implied)):
        target = np.maximum(np.asarray(implied, dtype=float) - UNMODELED_PTS, 3.0)
        for _ in range(rounds):
            s = Sim(G, L, calib_n, k, rng).run()
            mean = s.score.mean(0)
            k = k * np.clip(target / np.maximum(mean, 1.0), 0.7, 1.4) ** (1 / 1.6)
            k = np.clip(k, 0.55, 1.8)
    return Sim(G, L, n, k, rng).run(), k


def sim_quantiles(sim, levels, jitter_rec=True, seed=1):
    """Per player and prop: quantiles of the simulated stat (receptions jittered like the main model)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, col in STAT.items():
        x = sim.stats[:, :, col].astype(float)
        if k == "rec" and jitter_rec:
            x = x + rng.uniform(-0.5, 0.5, x.shape)
        out[k] = np.quantile(x, levels, axis=0).T  # (players, levels)
    return out


def sim_touchdowns(sim):
    """Per player: share of simulated games in which he scored a rushing or receiving touchdown (any), two or more
    (two), the game's first touchdown (first; a defensive or return TD first counts for nobody), and his average."""
    td = sim.stats[:, :, TD_RUSH] + sim.stats[:, :, TD_REC]
    first = np.bincount(sim.first_td[sim.first_td >= 0], minlength=sim.G["n"]) / sim.n
    return {"any": (td >= 1).mean(0), "two": (td >= 2).mean(0), "first": first, "mean": td.mean(0)}
