"""
Feature pipeline for NFL player-prop models.
Every feature for a player-game is built ONLY from games before it (shift(1)),
plus things known before kickoff: Vegas spread/total, home/away, weather/roof,
rest days, the announced starting QB, and who is active.
"""
import numpy as np
import pandas as pd

SKILL = ["QB", "RB", "WR", "TE", "FB"]

# Player stats use today's team codes for every season, but the schedule and snap counts still use the old
# ones (OAK 2016-19, SD 2016). Without this, Raiders/Chargers home games of those years were treated as road
# games, with the spread flipped and the opponent's QB marked as their starter.
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Per-game stats we roll forward as features
ROLL_STATS = [
    "targets", "receptions", "receiving_yards", "receiving_air_yards", "target_share",
    "air_yards_share", "carries", "rushing_yards", "attempts", "completions",
    "passing_yards", "passing_air_yards", "offense_pct", "carry_share",
    "rec_yards_gained_exp", "rush_yards_gained_exp", "pass_yards_gained_exp", "receptions_exp",
    "receiving_yards_after_catch", "sacks_suffered", "passing_epa", "receiving_epa", "rushing_epa",
]


def _seq(df):
    return df["season"] * 100 + df["week"]


# Games a player left early hurt are dropped from his usage averages (they read like a role he doesn't have).
MASK_INJURY_EXITS = True
HURT_STATUS = ("Questionable", "Doubtful", "Out")


def injury_exits(df, path="."):
    """True for player-games he likely left hurt: under half his usual snap share (median of his previous 5
    games, and him a regular at 40%+), and for his team's next game he either didn't play or was on the injury
    report as Questionable/Doubtful/Out or with limited/no practice. A snap drop with a clean report is a role
    change, so it stays in the averages. Uses only what's known before the game being predicted (the next game's
    report comes out before that game). df must be sorted by player_id, seq."""
    try:
        inj = pd.read_parquet(f"{path}/injuries.parquet")
    except FileNotFoundError:
        return pd.Series(False, index=df.index)
    prac = inj["practice_status"].fillna("")
    hurt = inj[inj["report_status"].isin(HURT_STATUS) | prac.str.contains("Did Not|Limited")]
    hurt_keys = set(zip(hurt["season"].astype(int), hurt["week"].astype(int), hurt["gsis_id"]))

    pid = df["player_id"]
    usual = (df["offense_pct"].groupby(pid).shift(1).groupby(pid).rolling(5, min_periods=2).median()
             .reset_index(level=0, drop=True).sort_index())
    low = (df["offense_pct"] < 0.5 * usual) & (usual >= 0.4) & (df.get("is_future", 0) != 1)

    tg = df[["team", "seq", "season", "week"]].drop_duplicates(["team", "seq"]).sort_values(["team", "seq"])
    nxt = tg.groupby("team")[["seq", "season", "week"]].shift(-1)
    tg = tg.assign(next_seq=nxt["seq"], next_season=nxt["season"], next_week=nxt["week"])
    m = df[["team", "seq"]].merge(tg, on=["team", "seq"], how="left")
    played = set(zip(df["player_id"], df["seq"]))
    has_next = m["next_seq"].notna().to_numpy()
    rows = zip(df["player_id"], m["next_seq"], m["next_season"], m["next_week"], has_next)
    flag = [h and ((p, ns) not in played or (int(s), int(w), p) in hurt_keys) for p, ns, s, w, h in rows]
    return low & pd.Series(flag, index=df.index)


def repair_starting_qbs(sched, ps):
    """For finished games, replace a listed starting QB who never threw a pass with the team's leading passer.

    nflverse sometimes lists the wrong starter (e.g. WAS 2024 weeks 9-12 list Mariota; Daniels threw every pass).
    A listed starter who threw at least once is kept even if he left early: that is a real outcome for his props.
    """
    q = ps[(ps["position"] == "QB") & (ps["attempts"] > 0)]
    threw = set(zip(q["game_id"], q["player_id"]))
    top = q.sort_values("attempts").groupby(["game_id", "team"]).tail(1)
    top = dict(zip(zip(top["game_id"], top["team"]), top["player_id"]))
    sched = sched.copy()
    played = sched["result"].notna() if "result" in sched else sched["home_score"].notna()
    for side in ("home", "away"):
        listed = list(zip(sched["game_id"], sched[f"{side}_qb_id"]))
        leader = pd.Series([top.get(k) for k in zip(sched["game_id"], sched[f"{side}_team"])], index=sched.index)
        wrong = played & pd.Series([k not in threw for k in listed], index=sched.index) & leader.notna()
        sched.loc[wrong, f"{side}_qb_id"] = leader[wrong]
    return sched


def game_weather(path="."):
    """game_id -> precip (rain, showers, drizzle, snow...) and snow flags, outdoor games only (game_weather.parquet,
    built by run.py's download from the play-by-play weather text)."""
    try:
        w = pd.read_parquet(f"{path}/game_weather.parquet")[["game_id", "precip", "snow"]]
    except FileNotFoundError:
        return pd.DataFrame(columns=["game_id", "precip", "snow"])
    return w.drop_duplicates("game_id")


def load_forecast(path="."):
    import json
    import os
    f = f"{path}/forecast.json"
    return json.load(open(f)) if os.path.exists(f) else {}


def qb_starts(sched):
    """Per (game, team): the starting QB's earlier starts for this team and in his career (any team), since 2016.
    New starters have 0-3 team starts; the model's QB-history features otherwise can't tell a QB's first start for a
    new team from his 50th."""
    rows = []
    for side in ("home", "away"):
        rows.append(sched[["game_id", "gameday", f"{side}_team", f"{side}_qb_id"]].set_axis(
            ["game_id", "gameday", "team", "qb"], axis=1))
    t = pd.concat(rows).dropna(subset=["qb"]).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    t["sqb_team_starts"] = t.groupby(["qb", "team"]).cumcount()
    t["sqb_starts"] = t.groupby("qb").cumcount()
    return t[["game_id", "team", "sqb_team_starts", "sqb_starts"]]


def load_base(path=".", extra_rows=None):
    ps = pd.read_parquet(f"{path}/ps.parquet")
    sc = pd.read_parquet(f"{path}/sc.parquet")
    players = pd.read_parquet(f"{path}/players.parquet")
    ffo = pd.read_parquet(f"{path}/ffopp.parquet")
    sched = pd.read_parquet(f"{path}/sched.parquet")
    sched[["home_team", "away_team"]] = sched[["home_team", "away_team"]].replace(TEAM_FIX)
    sc[["team", "opponent"]] = sc[["team", "opponent"]].replace(TEAM_FIX)

    ps = ps[ps["position"].isin(SKILL)].copy()
    sched = repair_starting_qbs(sched, ps)

    # --- snap counts -> gsis ids; include players who played snaps but logged no stats (true zeros)
    idmap = players.dropna(subset=["pfr_id", "gsis_id"])[["pfr_id", "gsis_id"]].drop_duplicates("pfr_id")
    sc = sc[sc["position"].isin(SKILL) & (sc["offense_snaps"] > 0)]
    sc = sc.merge(idmap, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    sc = sc.rename(columns={"gsis_id": "player_id"})[
        ["game_id", "player_id", "player", "position", "team", "opponent", "season", "week",
         "game_type", "offense_snaps", "offense_pct"]
    ].drop_duplicates(["game_id", "player_id"])

    df = ps.merge(sc[["game_id", "player_id", "offense_snaps", "offense_pct"]],
                  on=["game_id", "player_id"], how="outer", indicator=True)
    # fill identity columns for snap-only rows
    only_sc = df["_merge"] == "right_only"
    scx = sc.set_index(["game_id", "player_id"])
    idx = pd.MultiIndex.from_frame(df.loc[only_sc, ["game_id", "player_id"]])
    df.loc[only_sc, "player_display_name"] = scx.loc[idx, "player"].values
    df.loc[only_sc, "position"] = scx.loc[idx, "position"].values
    df.loc[only_sc, "team"] = scx.loc[idx, "team"].values
    df.loc[only_sc, "opponent_team"] = scx.loc[idx, "opponent"].values
    df.loc[only_sc, "season"] = scx.loc[idx, "season"].values
    df.loc[only_sc, "week"] = scx.loc[idx, "week"].values
    df.loc[only_sc, "season_type"] = np.where(scx.loc[idx, "game_type"].values == "REG", "REG", "POST")
    df = df.drop(columns="_merge")
    statcols = [c for c in ps.columns if ps[c].dtype.kind in "fi" and c not in ("season", "week")]
    if extra_rows is not None:  # future games: identity columns only, stats unknown
        df = pd.concat([df, extra_rows], ignore_index=True)
    df[statcols] = df[statcols].fillna(0)
    df = df.copy()  # consolidate memory blocks (avoids pandas "highly fragmented" warnings below)

    # --- expected-yards data
    fcols = ["rec_yards_gained_exp", "rush_yards_gained_exp", "pass_yards_gained_exp", "receptions_exp"]
    df = df.merge(ffo[["game_id", "player_id"] + fcols].drop_duplicates(["game_id", "player_id"]),
                  on=["game_id", "player_id"], how="left")
    df[fcols] = df[fcols].fillna(0)

    # --- schedule / Vegas / weather
    s = sched[["game_id", "gameday", "home_team", "away_team", "spread_line", "total_line", "roof",
               "temp", "wind", "home_rest", "away_rest", "home_qb_id", "away_qb_id", "div_game"]]
    df = df.merge(s, on="game_id", how="inner")
    retract = {"ATL", "IND", "HOU", "ARI", "DAL"}  # retractable roofs: assume closed when unknown
    s_roof_missing = df["roof"].isna() & df["home_team"].isin(retract)
    df.loc[s_roof_missing, "roof"] = "closed"
    # climatology for outdoor games with no weather yet: same home team, same part of season, past years
    hist = sched[(sched["roof"] == "outdoors") & sched["temp"].notna()].copy()
    hist["wk_bin"] = (hist["week"] // 4)
    clim = hist.groupby(["home_team", "wk_bin"])[["temp", "wind"]].mean()
    need = df["temp"].isna() & (df["roof"] == "outdoors")
    if need.any():
        keys = list(zip(df.loc[need, "home_team"], df.loc[need, "week"] // 4))
        vals = clim.reindex(keys)
        df.loc[need, "temp"] = vals["temp"].values
        df.loc[need, "wind"] = vals["wind"].values
    # rain/snow: from the play-by-play weather text for played games; for upcoming games from forecast.json
    # (run.py writes it from the National Weather Service forecast at kickoff), else assumed dry
    df = df.merge(game_weather(path), on="game_id", how="left")
    fc = load_forecast(path)
    for gid, f in fc.items():
        m = df["game_id"] == gid
        if m.any():
            for c in ("temp", "wind", "precip", "snow"):
                if f.get(c) is not None:
                    df.loc[m, c] = f[c]
    df[["precip", "snow"]] = df[["precip", "snow"]].fillna(0)
    df["is_home"] = (df["team"] == df["home_team"]).astype(int)
    df["team_spread"] = np.where(df["is_home"] == 1, df["spread_line"], -df["spread_line"])  # + = favored
    df["implied_total"] = df["total_line"] / 2 + df["team_spread"] / 2
    df["opp_implied_total"] = df["total_line"] - df["implied_total"]
    df["rest"] = np.where(df["is_home"] == 1, df["home_rest"], df["away_rest"])
    df["starting_qb"] = np.where(df["is_home"] == 1, df["home_qb_id"], df["away_qb_id"])
    df["is_starting_qb"] = (df["player_id"] == df["starting_qb"]).astype(int)
    df["dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    df["wind"] = np.where(df["dome"] == 1, 0, df["wind"])
    df["temp"] = np.where(df["dome"] == 1, 70, df["temp"])
    df["seq"] = _seq(df)
    df["is_post"] = (df["season_type"] == "POST").astype(int)
    df = df.merge(qb_starts(sched), on=["game_id", "team"], how="left")
    return df


def team_tables(df):
    """Team offense volume and defense-allowed tables, rolled forward per team."""
    off = df.groupby(["game_id", "team", "seq"], as_index=False).agg(
        team_att=("attempts", "sum"), team_carries=("carries", "sum"),
        team_sacks=("sacks_suffered", "sum"), team_pass_yds=("passing_yards", "sum"),
        team_rush_yds=("rushing_yards", "sum"), team_targets=("targets", "sum"))
    off["team_plays"] = off["team_att"] + off["team_carries"] + off["team_sacks"]
    off["team_pass_rate"] = (off["team_att"] + off["team_sacks"]) / off["team_plays"].clip(lower=1)
    off = off.sort_values(["team", "seq"])
    ocols = ["team_att", "team_carries", "team_plays", "team_pass_rate", "team_pass_yds", "team_rush_yds"]
    sh = off.groupby("team")[ocols].shift(1)
    for c in ocols:
        off[f"{c}_ewm"] = sh[c].groupby(off["team"]).ewm(halflife=4, min_periods=1).mean().reset_index(level=0, drop=True)

    # defense: what offenses did against each defense, split by position
    d = df.copy()
    d["def_team"] = d["opponent_team"]
    agg = {"passing_yards": "sum", "rushing_yards": "sum", "carries": "sum", "attempts": "sum",
           "sacks_suffered": "sum", "pass_yards_gained_exp": "sum"}
    de = d.groupby(["game_id", "def_team", "seq"], as_index=False).agg(agg)
    de = de.rename(columns={"passing_yards": "dal_pass_yds", "rushing_yards": "dal_rush_yds",
                            "carries": "dal_carries", "attempts": "dal_att", "sacks_suffered": "dal_sacks",
                            "pass_yards_gained_exp": "dal_pass_exp"})
    de["dal_ypa"] = de["dal_pass_yds"] / de["dal_att"].clip(lower=1)
    de["dal_ypc"] = de["dal_rush_yds"] / de["dal_carries"].clip(lower=1)
    de["dal_pass_over_exp"] = de["dal_pass_yds"] - de["dal_pass_exp"]
    pos = d[d["position"].isin(["WR", "TE", "RB"])].pivot_table(
        index=["game_id", "def_team"], columns="position",
        values=["receiving_yards", "receptions"], aggfunc="sum", fill_value=0)
    pos.columns = [f"dal_{a}_{b}" for a, b in pos.columns]
    de = de.merge(pos.reset_index(), on=["game_id", "def_team"], how="left").fillna(0)
    de = de.sort_values(["def_team", "seq"])
    dcols = [c for c in de.columns if c.startswith("dal_")]
    sh = de.groupby("def_team")[dcols].shift(1)
    for c in dcols:
        de[f"{c}_ewm"] = sh[c].groupby(de["def_team"]).ewm(halflife=6, min_periods=1).mean().reset_index(level=0, drop=True)
    return off, de


def build_features(df, mask_exits=None):
    df = df.sort_values(["player_id", "seq"]).reset_index(drop=True)

    # carry share within team-game
    tc = df.groupby(["game_id", "team"])["carries"].transform("sum")
    df["carry_share"] = df["carries"] / tc.clip(lower=1)

    # usage averages skip games he left hurt (targets stay as they happened)
    mask_exits = MASK_INJURY_EXITS if mask_exits is None else mask_exits
    exits = injury_exits(df) if mask_exits else pd.Series(False, index=df.index)
    df["injury_exit"] = exits.astype(int)
    base = df[ROLL_STATS].mask(exits)

    g = df.groupby("player_id")
    pid = df["player_id"]
    sh = base.groupby(pid).shift(1)
    ps_key = pid.astype(str) + "_" + df["season"].astype(str)
    new_cols = {}
    def _d(r):  # drop group level, realign to df index
        return r.reset_index(level=0, drop=True).sort_index()
    for c in ROLL_STATS:
        x = sh[c]
        gx = x.groupby(pid)
        new_cols[f"{c}_last"] = x.groupby(pid).ffill()  # last game he wasn't hurt in
        new_cols[f"{c}_ewm"] = _d(gx.ewm(halflife=3, min_periods=1).mean())
        new_cols[f"{c}_l5"] = _d(gx.rolling(5, min_periods=1).mean())
        new_cols[f"{c}_l5med"] = _d(gx.rolling(5, min_periods=1).median())
        new_cols[f"{c}_l16"] = _d(gx.rolling(16, min_periods=1).mean())
        new_cols[f"{c}_l16sd"] = _d(gx.rolling(16, min_periods=3).std())
        # season-to-date (current season only, prior games)
        cur = _d(base[c].groupby(ps_key).expanding().mean())
        new_cols[f"{c}_std"] = cur.groupby(ps_key).shift(1)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # efficiency over last 16 games (sums of prior games)
    def roll_sum(col, n=16):
        return _d(sh[col].groupby(pid).rolling(n, min_periods=1).sum())
    tg, rc, ry, ra = roll_sum("targets"), roll_sum("receptions"), roll_sum("receiving_yards"), roll_sum("rushing_yards")
    ca, at, py = roll_sum("carries"), roll_sum("attempts"), roll_sum("passing_yards")
    rx = roll_sum("rec_yards_gained_exp")
    df["ypt_l16"] = ry / tg.replace(0, np.nan)
    df["catch_rate_l16"] = rc / tg.replace(0, np.nan)
    df["ypc_l16"] = ra / ca.replace(0, np.nan)
    df["ypa_l16"] = py / at.replace(0, np.nan)
    df["rec_yds_over_exp_l16"] = (ry - rx) / tg.replace(0, np.nan)
    df["career_games"] = g.cumcount()
    df["season_games"] = df.groupby([pid, df["season"]]).cumcount()
    df["days_since_last"] = (pd.to_datetime(df["gameday"]) - pd.to_datetime(g["gameday"].shift(1))).dt.days
    df["team_changed"] = (df["team"] != g["team"].shift(1)).astype(int)

    # --- active-teammate context: how much prior target/carry share is active this week
    for c in ["target_share", "carry_share", "air_yards_share"]:
        tot = df.groupby(["game_id", "team"])[f"{c}_ewm"].transform("sum")
        df[f"team_active_{c}_sum"] = tot
        df[f"{c}_norm"] = df[f"{c}_ewm"] / tot.replace(0, np.nan)
        df[f"{c}_rank"] = df.groupby(["game_id", "team"])[f"{c}_ewm"].rank(ascending=False)

    # --- starting QB context for everyone on the team
    qb = df[["game_id", "player_id", "passing_yards_ewm", "attempts_ewm", "passing_epa_ewm",
             "career_games", "pass_yards_gained_exp_ewm"]].rename(columns={
        "player_id": "starting_qb", "passing_yards_ewm": "sqb_pass_yds_ewm", "attempts_ewm": "sqb_att_ewm",
        "passing_epa_ewm": "sqb_epa_ewm", "career_games": "sqb_career_games",
        "pass_yards_gained_exp_ewm": "sqb_pass_exp_ewm"})
    df = df.merge(qb, on=["game_id", "starting_qb"], how="left")
    tq = df[["team", "seq", "starting_qb"]].drop_duplicates(["team", "seq"]).sort_values(["team", "seq"])
    tq["qb_changed"] = (tq["starting_qb"] != tq.groupby("team")["starting_qb"].shift(1)).astype(int)
    df = df.merge(tq[["team", "seq", "qb_changed"]], on=["team", "seq"], how="left")

    # --- team volume & opponent defense
    off, de = team_tables(df)
    df = df.merge(off[["game_id", "team"] + [c for c in off.columns if c.endswith("_ewm")]],
                  on=["game_id", "team"], how="left")
    df = df.merge(de[["game_id", "def_team"] + [c for c in de.columns if c.endswith("_ewm")]].rename(
        columns={"def_team": "opponent_team"}), on=["game_id", "opponent_team"], how="left")

    # --- volume x efficiency projections (a tree can't multiply two features on its own)
    half = lambda a, b: 0.5 * (df[a] + df[b])  # noqa: E731  team offense vs. what this opponent allows
    proj = {
        "x_targets": df["target_share_norm"] * df["team_att_ewm"],
        "x_carries": df["carry_share_norm"] * df["team_carries_ewm"],
        "x_rec_yds_air": df["air_yards_share_norm"] * half("team_pass_yds_ewm", "dal_pass_yds_ewm"),
        "x_rush_matchup": df["carry_share_norm"] * half("team_rush_yds_ewm", "dal_rush_yds_ewm"),
        "x_pass_matchup": half("team_pass_yds_ewm", "dal_pass_yds_ewm"),
        "x_pass_vegas": df["passing_yards_ewm"] * df["implied_total"] / 22.0,
        "x_tgt_vegas": df["target_share_norm"] * df["implied_total"],
    }
    proj["x_receptions"] = proj["x_targets"] * df["catch_rate_l16"]
    proj["x_rec_yds"] = proj["x_targets"] * df["ypt_l16"]
    proj["x_rush_yds"] = proj["x_carries"] * df["ypc_l16"]
    df = pd.concat([df, pd.DataFrame(proj, index=df.index)], axis=1)
    df["pos_code"] = df["position"].map({"QB": 0, "RB": 1, "FB": 1, "WR": 2, "TE": 3})
    return df


FEATURES_EXTRA = [
    "is_home", "team_spread", "implied_total", "opp_implied_total", "total_line", "rest", "dome",
    "temp", "wind", "div_game", "is_post", "week", "pos_code", "is_starting_qb",
    "ypt_l16", "catch_rate_l16", "ypc_l16", "ypa_l16", "rec_yds_over_exp_l16",
    "career_games", "season_games", "days_since_last", "team_changed",
    "team_active_target_share_sum", "target_share_norm", "target_share_rank",
    "team_active_carry_share_sum", "carry_share_norm", "carry_share_rank",
    "team_active_air_yards_share_sum", "air_yards_share_norm",
    "sqb_pass_yds_ewm", "sqb_att_ewm", "sqb_epa_ewm", "sqb_career_games", "sqb_pass_exp_ewm", "qb_changed",
]
# Added 2026-11-12. Walk-forward 2024-26 (retrain every 4 weeks) vs. without them, CRPS: rushing -0.18% (SE 0.09),
# receiving yards -0.07% (0.06), receptions -0.10% (0.07); in rain games -0.5% to -1.1% for every stat. Passing:
# +0.05% (0.16) with all four, 0.0% with rain only, so passing keeps the old set; the touchdown model (market None)
# wasn't re-backtested with them, so it keeps the old set too. The QB-start counts did NOT fix the new-starter
# passing bias (2026 new starters still -38 yds vs median), so the page's -20 yd verdict shade stays.
NEW_FEATURES = ["precip", "snow", "sqb_team_starts", "sqb_starts"]
NEW_FEATURE_MARKETS = {"rushing_yards", "receiving_yards", "receptions"}
USE_NEW = True  # set False to reproduce the pre-2026-11-12 feature set (backtest comparison)


PROJECTIONS = ["x_targets", "x_receptions", "x_rec_yds", "x_rec_yds_air", "x_carries", "x_rush_yds",
               "x_rush_matchup", "x_pass_matchup", "x_pass_vegas", "x_tgt_vegas"]
# In the 2024-26 walk-forward backtest these helped receiving yards in every season (CRPS -0.1% to -0.4%),
# did nothing for receptions and made rushing and passing yards worse, so only receiving yards uses them.
PROJECTION_MARKETS = {"receiving_yards"}


def feature_list(df, market=None):
    roll = [c for c in df.columns if any(c == f"{s}_{k}" for s in ROLL_STATS
                                          for k in ["last", "ewm", "l5", "l5med", "l16", "l16sd", "std"])]
    team = [c for c in df.columns if (c.startswith("team_") or c.startswith("dal_")) and c.endswith("_ewm")]
    proj = PROJECTIONS if market in PROJECTION_MARKETS else []
    new = [c for c in NEW_FEATURES if c in df.columns] if USE_NEW and market in NEW_FEATURE_MARKETS else []
    return roll + team + FEATURES_EXTRA + new + proj


if __name__ == "__main__":
    import time
    t = time.time()
    base = load_base()
    print("base", base.shape, time.time() - t)  # (run.py is the normal entry point)
    feat = build_features(base)
    print("features", feat.shape, len(feature_list(feat, "receiving_yards")), time.time() - t)
    feat.to_parquet("features.parquet")
