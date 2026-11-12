"""
Builds the expected-to-play rows for an upcoming week.
Expected-to-play = active-roster skill players who have played this season, minus anyone
ruled Out/Doubtful, minus players who missed their team's last 2 games and either didn't practice or
weren't on a recent injury report (healthy scratches / demotions). Questionable, limited-practice,
did-not-practice, returning and newly traded players are kept but flagged.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import nflreadpy as nfl

SKILL = ["QB", "RB", "WR", "TE", "FB"]
EASTERN = ZoneInfo("America/New_York")  # schedule kickoff times are Eastern


def kicked_off(sched, now=None):
    """True for games whose listed kickoff has passed: their projections stay frozen at the last pre-game run."""
    now = now or datetime.now(EASTERN)
    t = pd.to_datetime(sched["gameday"].astype(str) + " " + sched["gametime"].fillna("23:59").astype(str),
                       errors="coerce")
    return pd.Series(t.dt.tz_localize(EASTERN, ambiguous="NaT", nonexistent="shift_forward") <= now,
                     index=sched.index)


def last_starters(ps, season):
    """Team -> player_id of the QB with the most attempts in the team's most recent game."""
    q = ps[(ps.season == season) & (ps.position == "QB") & (ps.attempts > 0)]
    q = q.sort_values(["week", "attempts"]).groupby("team").tail(1)
    return dict(zip(q.team, q.player_id)), dict(zip(q.team, q.player_display_name))


def future_rows(season, week, sched, ps):
    games = sched[(sched.season == season) & (sched.week == week) & sched.home_score.isna() & ~kicked_off(sched)]
    ps = ps[(ps.season == season) & ps.position.isin(SKILL)]
    roster = nfl.load_rosters_weekly([season]).to_pandas()
    roster = roster[(roster.week == week) & (roster.status == "ACT") & roster.position.isin(SKILL)]
    if roster.empty:
        raise SystemExit(f"No week {week} rosters published yet. Try again closer to game day.")
    inj = nfl.load_injuries([season]).to_pandas()
    recent_inj = set(inj[inj.week.isin([week - 1, week - 2, week - 3]) &
                         inj.report_status.isin(["Out", "Doubtful"])].gsis_id)
    inj = inj[inj.week == week].drop_duplicates("gsis_id", keep="last").set_index("gsis_id")

    played = set(ps.player_id)
    weeks_played = ps.groupby("player_id")["week"].agg(set).to_dict()
    last_team = ps.sort_values("week").groupby("player_id")["team"].last().to_dict()
    rows, flags = [], {}
    for _, g in games.iterrows():
        for team, opp, qb in [(g.away_team, g.home_team, g.away_qb_id), (g.home_team, g.away_team, g.home_qb_id)]:
            team_weeks = sorted(ps[ps.team == team].week.unique())
            last2 = set(team_weeks[-2:])
            for _, p in roster[roster.team == team].iterrows():
                pid = p.gsis_id
                is_qb_start = pid == qb
                if p.position == "QB" and not is_qb_start:
                    continue  # only the announced starter gets QB rows
                if pid not in played and not is_qb_start:
                    continue
                st = inj.loc[pid] if pid in inj.index else None
                rep = st.report_status if st is not None else None
                prac = st.practice_status if st is not None else None
                if rep in ("Out", "Doubtful"):
                    continue
                # games for any team count, so a player traded this week isn't mistaken for a healthy scratch
                missed_recent = len(weeks_played.get(pid, set()) & last2) == 0
                new_team = pid in last_team and last_team[pid] != team
                dnp = isinstance(prac, str) and "Did Not" in prac
                if missed_recent and not is_qb_start and (dnp or pid not in recent_inj):
                    continue
                flag = ""
                if rep == "Questionable":
                    flag = "Questionable"
                elif dnp:
                    flag = "Missed practice"
                elif isinstance(prac, str) and "Limited" in prac:
                    flag = "Limited practice"
                if missed_recent and pid in played and not flag:
                    flag = "Returning"
                if new_team and not flag:
                    flag = "New team"
                flags[pid] = flag
                rows.append(dict(game_id=g.game_id, player_id=pid, player_display_name=p.full_name,
                                 position=p.position, team=team, opponent_team=opp, season=season,
                                 week=week, season_type="REG", is_future=1))
    return pd.DataFrame(rows).drop_duplicates(["game_id", "player_id"]), flags
