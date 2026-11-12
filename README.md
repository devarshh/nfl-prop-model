# NFL player-prop projections

Predicts passing yards, rushing yards, receiving yards, receptions and touchdown
chances for every likely player in the upcoming NFL week, then builds a page where
you type a sportsbook line (or touchdown odds) and see the model's probability.

## One-time setup (PyCharm)
1. File > Open, choose this `nfl_prop_model` folder.
2. When PyCharm asks, create a virtual environment from `requirements.txt`
   (Python 3.10 or newer). Or in PyCharm's Terminal run:
       pip install -r requirements.txt
3. Mac only: if you get an error mentioning `libomp`, run `brew install libomp`
   (LightGBM needs it), then try again.

## Every week
    python run.py
(or right-click `run.py` > Run). It downloads fresh data, finds the next unplayed
week, trains the stats models, simulates every game 1,000 times, blends the two, and
opens `output/weekN_projections.html` in your browser. Takes about a minute or two.

## Two methods, blended
- Stats model (model.py): LightGBM quantile models trained on every player-game since
  2016 (usage, efficiency, matchup, Vegas lines, weather, who's active).
- Game simulation (simulate.py): plays each game snap by snap 1,000 times. League
  behaviour is learned from nflverse play-by-play (pass/run calls by down, distance,
  field position, score and clock; 4th downs; depth of target, completion and
  interception odds; yards after catch; runs, scrambles, sacks, penalties, punts,
  kickoffs, field goals, the clock). Each team brings its own tendencies (pass rate
  over expected, tempo, 4th-down aggressiveness, its QB's sack/scramble/accuracy/
  interception profile, its defense), each player his usage and efficiency, with
  game-to-game swings in usage and the chance of leaving a game early. Each game is
  tuned so its average score matches the betting market's implied team totals.
- The page and PDF show the blend, plus both methods' own numbers. Typing a line
  shows each method's probability and whether they agree on the side.

The weights live in blend.json (simulation weight per stat, plus a scale correction
for the simulation's overall level, which drifts from season to season). In the
2025-26 walk-forward backtest the blend beat the stats model alone for every stat.
Refresh blend.json every few weeks (about 30 minutes, mostly backtest.py):
    python features.py
    python backtest.py --season 2025          # once; saves backtest_2025.parquet
    python backtest.py                         # this season so far
    python backtest_sim.py 2025 2026           # simulates those games (~5 min)
    python blend.py                            # writes blend.json

## Touchdowns
Each player's chance to score at least one rushing or receiving touchdown ("anytime";
passing TDs, kick/punt returns and 2-point conversions don't count), to score 2+, and to
score the game's first touchdown.
- Stats model (touchdowns.py): a LightGBM Poisson model of touchdowns per game, from the
  player's touchdowns and expected touchdowns (his red-zone and goal-line role), his share
  of what the active players usually get, team implied points, team scoring and what the
  opponent allows.
- Game simulation: the share of the 1,000 simulated games in which he scored. Near the goal
  line the simulator hands the ball to the players who usually get it there (short-yardage
  backs, sneaking QBs, tight ends), and the first-TD chance comes from it alone.
- The page lists each game's likeliest scorers; type a book's odds (+150, -120 or 2.50) in a
  player's "Anytime TD" row to see the expected value of a $1 bet and whether both methods
  agree. Books take a big margin on touchdown markets, so most prices show no value.

Refresh the touchdown blend weight along with the yardage one (a few minutes):
    python touchdowns.py backtest 2025 2026    # stats model, week by week
    python backtest_sim.py 2025 2026           # (also needed for blend.py)
    python touchdowns.py blend                 # writes blend.json["td"]

## Parlays
The page's Parlays tab (and each game's card) shows the most likely parlay built from the
standard "X+" milestone bets (50+ rushing yards, 5+ receptions, 250+ passing yards...) and
touchdown bets: within each game (2-5 legs, one per player unless you allow stacking) and across
games (one leg per game). Each leg is capped at 70%, 80% or 90% likely on its own, since an
uncapped "most likely" parlay just stacks near-certain legs that pay nothing. Leg chances are
the blended projections; how often legs hit together comes from the 1,000 simulated games
(parlays.py). Type a book's parlay price to see its expected value.
Backtest (406 games, 2025-26): suggested 3-5-leg parlays hit within ~2 points of their predicted
chance at the 70%/80% caps; 2-leg ones ~4 points less often; 90%-cap ones ~5 points more often.
    python parlays.py backtest 2025 2026 <stats-model backtest .parquet files>

Best times to run: Wednesday/Thursday for an early look, then again Friday evening
after final injury designations (or Sunday morning). Lines you type into the page are
saved in your browser, per week.

Options:
    python run.py --week 10       # a specific week
    python run.py --no-download   # reuse already-downloaded data
    python run.py --qb MIN="Carson Wentz"   # the schedule still lists an injured starter
                                            # (run.py prints a !! warning when a listed starter is Out)
    python run.py --out "George Pickens"    # inactive / ruled out after the injury report (repeatable):
                                            # his targets and carries go to the teammates who play

Inactives: about 90 minutes before kickoff each team posts its inactive list. Rerun with --out for any
projected player on it (the models already hand a missing player's share to his active teammates: every
usage feature is his share divided by the active players' total).

Weather: run.py fetches the National Weather Service hourly forecast for open-air games (kickoff to +3 h)
into forecast.json: temperature, wind and rain/snow (50%+ chance). Played games' rain/snow comes from the
play-by-play weather text (game_weather.parquet). Rain games since 2016 finished 3.2 points under the
closing total (167 games, SE 1.0) with less passing (pass rate over expected -3.3 pts, completion % over
expected -3.5 pts): the simulator applies that scaled by the rain chance, and the stats model has rain and
snow as features.

Bet / Lean / Pass: type the book's line plus over and under odds (blank = -110) under any prop, or its
price on a touchdown or parlay, and the page gives a verdict with its reasons: edge over the price's
break-even (Bet 5+ pts, Lean 2.5+; TDs and parlays by EV: Bet +10c, Lean +3c per $1), both methods on the
same side, caps for Questionable / missed practice / new team / tail lines / 75%+ gaps, and two backtest
bias fixes applied to the verdict only (new-starter QB passing -20 yds; goal-line QB TDs lean to the stats
model). Thresholds live in RULES at the top of week_template.html's script. "Download my bets (CSV)"
before kickoff, then grade it after the games:
    python review_week.py 9 --bets ~/Downloads/week9_bets.csv

Injury exits and role changes: games a player left hurt (under half his usual snaps, then missed the next
game or was listed Questionable/Doubtful/Out or limited in practice) are left out of his usage averages
(features.py, injuries.parquet). Backtest 2024-26: receiving yards and receptions about 0.2% better, ~1% for
players with a recent exit; passing and rushing unchanged. Players whose snap share moved 15+ points in their
last two games get a "Snaps" tag on the page and PDF; a drop caps the verdict at Lean.

Reality check after games (grades every final game's props, TDs and parlays):
    python run.py                 # attaches box scores to finished games
    python review_week.py 9       # calibration, misses, DNPs; writes output/week9_graded.csv

Printable version (after run.py; needs Google Chrome installed):
    python make_pdf.py            # writes output/weekN_projections.pdf

Games that have kicked off keep their last pre-game projections. Once nflverse has the
box score (usually the next morning), rerunning run.py attaches the actual results, and
the PDF shows how each finished game's props did next to the upcoming games.

## Things to check yourself
- The script prints any QB changes from last week. Starting QBs come from the
  nflverse schedule; confirm them in the news.
- Flags on the page: Questionable, Missed practice, Limited practice, Returning,
  New starter, New team. Projections for teammates assume flagged players play.
- Weather: forecasts only within about a week of the game; earlier runs use the stadium's typical conditions.

## Optional: re-run the backtest
    python features.py            # builds features.parquet
    python model.py               # optional: re-tune tree counts into iters.json (~5 min)
    python backtest.py            # every played week of the current season (or: python backtest.py 1 2 3)
    python evaluate.py

## Files
- run.py: the one command
- features.py: feature engineering (pre-game info only)
- predict_week.py: who's expected to play
- model.py: LightGBM quantile models (5th to 95th percentile) and the over/under math;
  iters.json holds tuned tree counts
- week_template.html: the projections page
- make_pdf.py: printable PDF of the latest projections
- simulate.py: the play-by-play game simulator
- backtest_sim.py, blend.py, blend.json: simulator backtest and the blend weights
- touchdowns.py: the touchdown model, its backtest and its blend weight
- parlays.py: most likely parlays (same game and across games) and their backtest
