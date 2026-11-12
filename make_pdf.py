"""
Printable PDF of a week's projections. Run it after run.py:

    python make_pdf.py              # latest week in output/
    python make_pdf.py --week 9     # a specific week

Writes output/weekN_projections.pdf. Needs Google Chrome (or Chromium / Edge) installed; it prints the
page headlessly, so no extra Python packages are required.
"""
import argparse
import datetime as dt
import glob
import html
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from model import QUANTILES, cdf  # noqa: E402
import parlays  # noqa: E402

MARKETS = [("pass", "Passing yards"), ("rush", "Rushing yards"), ("recyd", "Receiving yards"),
           ("rec", "Receptions")]
SCALE = {"pass": (50, 400, 50), "rush": (0, 160, 20), "recyd": (0, 160, 20), "rec": (0, 12, 2)}  # lo, hi, grid
IDX = {q: QUANTILES.index(q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
BAR_W, BAR_H = 230, 12
RESULT_BAR_W = 125  # results tables have two more columns
TWO_BAR_W = 165     # upcoming tables with both methods' medians
TD_BAR_W, TD_MAX = 150, 0.8  # touchdown bars: 0-80% chance
WARN = {"Questionable", "Missed practice", "Limited practice", "New team", "New starter"}

CSS = """
@page { size: letter; margin: 0.45in 0.5in 0.5in; }
:root { --ink: #142019; --muted: #5A685E; --rule: #D3DBD2; --turf: #2C6A3D; --band: #CBD7CD;
        --band-strong: #8FA895; --slate: #5F6B78; --warn: #7A6200; --chip: #EEF2EC; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; color: var(--ink); background: #fff; font: 9pt/1.35 "Archivo", "Helvetica Neue", Arial, sans-serif;
       font-variation-settings: "wdth" 100; }
h1 { margin: 0 0 2px; font-size: 30pt; line-height: 1; font-weight: 800; font-variation-settings: "wdth" 66; }
.sub { color: var(--muted); font-size: 10pt; margin: 0 0 14px; }
.howto { display: grid; grid-template-columns: 250px 1fr; gap: 6px 20px; align-items: start;
         border-top: 2px solid var(--ink); border-bottom: 1px solid var(--rule); padding: 10px 0 12px; }
.howto h3, .notes h3, .slate h3 { margin: 0 0 4px; font-size: 11pt; font-variation-settings: "wdth" 80; }
.howto p, .notes p { margin: 0 0 5px; }
.howto svg text { font-size: 7.5pt; fill: var(--muted); }
.slate { margin-top: 14px; }
.slate table td, .slate table th { padding: 2px 6px 2px 0; }
.notes { margin-top: 12px; color: var(--ink); }
.notes p { color: var(--muted); }
.notes b { color: var(--ink); font-weight: 650; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; font-weight: 600; color: var(--muted); font-size: 7.5pt; text-transform: uppercase;
     letter-spacing: .04em; border-bottom: 1px solid var(--ink); padding: 2px 6px 3px 0; vertical-align: bottom; }
td { padding: 2.5px 6px 2.5px 0; border-bottom: 1px solid var(--rule); vertical-align: middle; }
tr { break-inside: avoid; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.game { margin-top: 18px; }
.keep { break-inside: avoid; }
.gh { border-top: 3px solid var(--ink); padding-top: 5px; margin-bottom: 2px; }
.gh h2 { margin: 0; font-size: 20pt; line-height: 1.05; font-weight: 800; font-variation-settings: "wdth" 70;
         display: flex; justify-content: space-between; align-items: baseline; }
.gh h2 small { font-size: 12pt; font-weight: 500; color: var(--muted); margin: 0 4px; }
.gh h2 .when { font-size: 10pt; font-weight: 600; font-variation-settings: "wdth" 85; }
.gh .meta { color: var(--muted); font-size: 8.5pt; }
.mk { break-inside: avoid; margin-top: 8px; }
.mk th.label { color: var(--turf); font-size: 8.5pt; font-weight: 700; }
.mk th.label .vs { color: var(--muted); font-weight: 500; font-size: 7pt; margin-left: 8px; letter-spacing: .02em; }
.games-start { break-before: page; }
.page2 { break-before: page; }
.mk td.nm { width: 245px; }
.mk .who { color: var(--muted); font-size: 7.5pt; margin-left: 4px; }
.mk td.med { font-weight: 700; font-size: 10.5pt; width: 50px; font-variation-settings: "wdth" 80; }
.mk td.rng { width: 78px; color: var(--ink); }
.mk td.bar, .mk th.bar { padding-right: 0; text-align: right; white-space: nowrap; }
.mk table.final td.nm { width: 250px; }
.mk table.final td.rng { width: 70px; }
.mk th.bar svg text { font-size: 6.5pt; fill: var(--muted); text-transform: none; letter-spacing: 0; }
.flag { font-size: 6.5pt; padding: 0 4px; border-radius: 6px; border: 1px solid var(--slate); color: var(--slate);
        margin-left: 4px; white-space: nowrap; vertical-align: 1px; }
.flag.warn { border-color: var(--warn); color: var(--warn); }
svg .g { stroke: #E6EBE4; stroke-width: 1; }
svg .b80 { fill: var(--band); }
svg .b50 { fill: var(--band-strong); }
svg .md { fill: var(--ink); }
svg .act { fill: #F2C200; stroke: #7A6200; stroke-width: 1; }
.mk td.act { font-weight: 700; font-size: 10.5pt; width: 46px; font-variation-settings: "wdth" 80; }
.mk td.pct { width: 52px; color: var(--muted); }
.mk td.two { width: 74px; color: var(--muted); font-size: 8pt; }
.mk td.two.apart { color: var(--warn); font-weight: 600; }
.mk td.pct.far { color: var(--warn); font-weight: 600; }
.gh .final { color: var(--turf); }
.mk td.tdp { font-weight: 700; font-size: 10.5pt; width: 44px; font-variation-settings: "wdth" 80; }
.mk td.odds { width: 46px; }
.mk td.sm { width: 46px; color: var(--muted); }
.tdcap { margin: 3px 0 0; color: var(--muted); font-size: 8pt; }
.toptd { columns: 2; column-gap: 24px; margin: 4px 0 0; padding: 0; list-style: none; }
.toptd li { break-inside: avoid; display: flex; justify-content: space-between; border-bottom: 1px solid var(--rule);
            padding: 1.5px 0; }
.toptd .who { color: var(--muted); font-size: 7.5pt; margin-left: 4px; }
.gh .parl { color: var(--ink); margin-top: 2px; }
.plist { margin: 4px 0 0; padding-left: 16px; }
.plist li { margin: 0 0 3px; }
.plist .who { color: var(--muted); font-size: 7.5pt; }
.score { margin: 4px 0 0; font-size: 8.5pt; }
.score b { font-weight: 650; }
.results { margin-top: 12px; }
.results p { margin: 0 0 5px; color: var(--muted); }
.results b { color: var(--ink); font-weight: 650; }
"""


def latest_week(week=None):
    files = glob.glob("output/week*.json")
    if week is not None:
        path = f"output/week{week}.json"
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run: python run.py --week {week}")
        return path
    if not files:
        raise SystemExit("No projections in output/. Run python run.py first.")
    return max(files, key=os.path.getmtime)


def fmt(v, k):
    v = max(v, 0.0)  # receptions are modeled with +-0.5 jitter; never show a negative count
    return f"{v:.1f}" if k == "rec" else f"{v:.0f}"


def kickoff(g, with_date=False):
    d = dt.date.fromisoformat(g["day"])
    day = d.strftime("%a") + (f" {d.strftime('%b')} {d.day}" if with_date else "")
    if not g.get("time"):
        return day
    h, m = map(int, g["time"].split(":"))
    return f"{day} {h % 12 or 12}:{m:02d} {'PM' if h >= 12 else 'AM'} ET"


def spread_text(g, short=False):
    if g.get("spread") is None:
        return "—" if short else "line not posted yet"
    s = g["spread"]
    fav, pts = (g["home"], s) if s > 0 else (g["away"], -s)
    if s == 0:
        return "Pick" if short else "pick'em"
    return f"{fav} −{pts:g}" if short else f"{fav} favored by {pts:g}"


def x_of(v, k, w=BAR_W):
    lo, hi, _ = SCALE[k]
    return (min(max(v, lo), hi) - lo) / (hi - lo) * w


def grid(k, w=BAR_W):
    lo, hi, step = SCALE[k]
    return "".join(f'<line class="g" x1="{x_of(t, k, w):.1f}" x2="{x_of(t, k, w):.1f}" y1="0" y2="{BAR_H}"/>'
                   for t in range(lo, hi + 1, step))


def bar(q, k, actual=None, w=BAR_W):
    a, b = x_of(q[IDX[0.1]], k, w), x_of(q[IDX[0.9]], k, w)
    c, d = x_of(q[IDX[0.25]], k, w), x_of(q[IDX[0.75]], k, w)
    m = x_of(q[IDX[0.5]], k, w)
    act = ("" if actual is None else  # yellow marker = what actually happened
           f'<rect class="act" x="{min(max(x_of(actual, k, w) - 2, 0.5), w - 4.5):.1f}" y="0.5" width="4" '
           f'height="{BAR_H - 1}"/>')
    return (f'<svg width="{w}" height="{BAR_H}" viewBox="0 0 {w} {BAR_H}">{grid(k, w)}'
            f'<rect class="b80" x="{a:.1f}" y="2.5" width="{max(b - a, 1):.1f}" height="7"/>'
            f'<rect class="b50" x="{c:.1f}" y="2.5" width="{max(d - c, 1):.1f}" height="7"/>'
            f'<rect class="md" x="{m - 1:.1f}" y="0" width="2" height="{BAR_H}"/>{act}</svg>')


def percentile(q, y, k):
    """Where the actual result landed in the predicted distribution, 0-100. Receptions are whole numbers, so
    use the midpoint of the probability mass at that count."""
    Q = np.array([q], dtype=float)
    if k == "rec":
        return 50 * (cdf(Q, y - 0.5)[0] + cdf(Q, y + 0.5)[0])
    return 100 * cdf(Q, y)[0]


def ordinal(p):
    if p < 1:
        return "<1st"
    if p > 99:
        return ">99th"
    n = int(round(p))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def graded(games):
    """One row per prop of every final game: (game, player, market, quantiles, actual, percentile)."""
    out = []
    for g in games:
        if not g.get("final"):
            continue
        for p in g["players"]:
            for k, q in p["m"].items():
                if p.get("a") and k in p["a"]:
                    y = p["a"][k]
                    out.append(dict(g=g, p=p, k=k, q=q, y=y, pct=percentile(q, y, k)))
    return out


def scorecard(rows):
    n = len(rows)
    in50 = sum(r["q"][IDX[0.25]] <= r["y"] <= r["q"][IDX[0.75]] for r in rows)
    in80 = sum(r["q"][IDX[0.1]] <= r["y"] <= r["q"][IDX[0.9]] for r in rows)
    over = sum(r["y"] > r["q"][IDX[0.5]] for r in rows)
    return (f'{n} props: <b>{in50} landed inside the middle 50%</b> (about {n * 0.5:.0f} expected), '
            f'<b>{in80} inside the middle 80%</b> (about {n * 0.8:.0f} expected); {over} went over the median, '
            f'{n - over} under.')


def axis(k, w=BAR_W):
    lo, hi, step = SCALE[k]
    ticks = []
    for t in range(lo, hi + 1, step if w >= 200 else 2 * step):  # fewer labels on the narrow results bar
        anchor = "start" if t == lo else "end" if t == hi else "middle"
        ticks.append(f'<text x="{x_of(t, k, w):.1f}" y="8" text-anchor="{anchor}">{t}</text>')
    return f'<svg width="{w}" height="10" viewBox="0 0 {w} 10">{"".join(ticks)}</svg>'


def flags(p):
    out = []
    if p.get("new"):
        out.append('<span class="flag warn">New starter</span>')
    if p.get("f"):
        out.append(f'<span class="flag{" warn" if p["f"] in WARN else ""}">{html.escape(p["f"])}</span>')
    if p.get("rc"):  # snap share moved a lot in his last 2 games (run.py role_changes)
        a, b, k = (p["rc"] + ["Snaps"])[:3]
        out.append(f'<span class="flag warn">{k}</span>' if k.startswith("Teammate") else
                   f'<span class="flag warn">{k} {"↓" if b < a else "↑"} {a:.0%}→{b:.0%}</span>')
    return "".join(out)


def market_table(g, k, label):
    rows = [(p, p["m"][k]) for p in g["players"] if k in p["m"]]
    if not rows:
        return ""
    rows.sort(key=lambda r: -r[1][IDX[0.5]])
    final = bool(g.get("final"))
    two = (not final) and any(k in p.get("s", {}) for p, _ in rows)  # both methods' medians available
    w = RESULT_BAR_W if final else (TWO_BAR_W if two else BAR_W)
    body = []
    for p, q in rows:
        y = p["a"].get(k) if final and p.get("a") else None
        cells = ""
        if final:
            pct = percentile(q, y, k) if y is not None else None
            far = pct is not None and (pct < 10 or pct > 90)
            cells = (f'<td class="num act">{fmt(y, k) if y is not None else "DNP"}</td>'
                     f'<td class="num pct{" far" if far else ""}">{ordinal(pct) if pct is not None else ""}</td>')
        elif two:
            if k in p.get("s", {}):
                gm, sm = p["g"][k][IDX[0.5]], p["s"][k][IDX[0.5]]
                apart = abs(sm - gm) >= 0.2 * max(abs(gm), 1) and abs(sm - gm) >= (0.8 if k == "rec" else 8)
                cells = f'<td class="num two{" apart" if apart else ""}">{fmt(gm, k)} · {fmt(sm, k)}</td>'
            else:
                cells = '<td class="num two"></td>'
        body.append(
            f'<tr><td class="nm"><b>{html.escape(p["n"])}</b><span class="who">{p["t"]} {p["p"]}</span>{flags(p)}</td>'
            f'<td class="num med">{fmt(q[IDX[0.5]], k)}</td>'
            f'<td class="num rng">{fmt(q[IDX[0.25]], k)}–{fmt(q[IDX[0.75]], k)}</td>'
            f'<td class="num rng">{fmt(q[IDX[0.1]], k)}–{fmt(q[IDX[0.9]], k)}</td>{cells}'
            f'<td class="bar">{bar(q, k, y, w)}</td></tr>')
    extra = ('<th class="num">Actual</th><th class="num">Pctile</th>' if final else
             '<th class="num">Stats · Sim</th>' if two else "")
    return (f'<table{" class=final" if final else ""}><thead><tr><th class="label">{label}'
            f'<span class="vs">{g["away"]} at {g["home"]}</span></th>'
            f'<th class="num">Median</th>'
            f'<th class="num">Middle 50%</th><th class="num">Middle 80%</th>{extra}<th class="bar">{axis(k, w)}</th>'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table>')


def pc(p):
    return "<1%" if p < 0.005 else f"{max(p, 0.01):.0%}"


def american(p):
    """Break-even American odds for probability p."""
    p = min(max(p, 0.01), 0.98)
    return f"−{100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def td_bar(p, w=TD_BAR_W):
    x = min(max(p, 0.0), TD_MAX) / TD_MAX * w
    grid_ = "".join(f'<line class="g" x1="{t / TD_MAX * w:.1f}" x2="{t / TD_MAX * w:.1f}" y1="0" y2="{BAR_H}"/>'
                    for t in (0.2, 0.4, 0.6))
    return (f'<svg width="{w}" height="{BAR_H}" viewBox="0 0 {w} {BAR_H}">{grid_}'
            f'<rect class="b50" x="0" y="2.5" width="{max(x, 1):.1f}" height="7"/></svg>')


def td_axis(w=TD_BAR_W):
    ticks = "".join(f'<text x="{t / TD_MAX * w:.1f}" y="8" text-anchor="{"start" if t == 0 else "end" if t == TD_MAX else "middle"}">'
                    f'{t:.0%}</text>' for t in (0, 0.2, 0.4, 0.6, 0.8))
    return f'<svg width="{w}" height="10" viewBox="0 0 {w} 10">{ticks}</svg>'


def td_table(g):
    """Anytime-touchdown chances for every listed player (blend; both methods; 2+; first TD from the sim)."""
    rows = sorted((p for p in g["players"] if p.get("td")), key=lambda p: -p["td"]["m"][0])
    if not rows:
        return ""
    final = bool(g.get("final"))
    body = []
    for p in rows:
        td = p["td"]
        a, two = td["m"]
        if "s" in td:
            apart = abs(td["g"][0] - td["s"][0]) >= 0.08
            gs = f'<td class="num two{" apart" if apart else ""}">{pc(td["g"][0])} · {pc(td["s"][0])}</td>'
        else:
            gs = '<td class="num two"></td>'
        act = ""
        if final:
            y = (p.get("a") or {}).get("td")
            act = f'<td class="num act">{"DNP" if p.get("a") is None else f"{y:.0f}" if y else "–"}</td>'
        first = td.get("first")
        body.append(
            f'<tr><td class="nm"><b>{html.escape(p["n"])}</b><span class="who">{p["t"]} {p["p"]}</span>{flags(p)}</td>'
            f'<td class="num tdp">{pc(a)}</td><td class="num odds">{american(a)}</td>'
            f'<td class="num sm">{pc(two)}</td><td class="num sm">{pc(first) if first is not None else ""}</td>'
            f'{gs}{act}<td class="bar">{td_bar(a)}</td></tr>')
    cap = ""
    if final:
        scored = [p for p in rows if (p.get("a") or {}).get("td")]
        cap = (f'<p class="tdcap">{len(scored)} of these players scored; the projections expected '
               f'{sum(p["td"]["m"][0] for p in rows):.1f}.</p>')
    return (f'<table><thead><tr><th class="label">Anytime touchdown<span class="vs">{g["away"]} at {g["home"]}</span></th>'
            f'<th class="num">Chance</th><th class="num">Fair</th><th class="num">2+ TDs</th><th class="num">First TD</th>'
            f'<th class="num">Stats · Sim</th>{"<th class=num>TDs</th>" if final else ""}'
            f'<th class="bar">{td_axis()}</th></tr></thead><tbody>{"".join(body)}</tbody></table>{cap}')


def leg_text(lg, game=False):
    where = f', {lg["game"]}' if game else ""
    return (f'{html.escape(lg["n"])} {lg["label"]} <span class="who">({lg["t"]} {lg["pos"]}{where})</span> '
            f'{pc(lg["p"])}')


def parlay_line(g, k=3, cap=0.8):
    """The game's most likely k-leg same-game parlay (each leg at most cap), for its header."""
    r = (g.get("parlays") or {}).get(f"{k}|{cap}")
    if not r or g.get("started"):
        return ""
    legs = " + ".join(f'{html.escape(lg["n"])} {lg["label"]} ({pc(lg["p"])})' for lg in r["legs"])
    return (f'<div class="meta parl"><b>Most likely parlay</b> ({k} players, each leg at most {cap:.0%}): {legs} → '
            f'<b>{pc(r["p"])} together</b>, fair {american(r["p"])} (multiplying the legs: {pc(r["ind"])})</div>')


def parlay_section(upcoming, cap=0.8):
    """Page-2 summary: the most likely parlay across games, and the week's likeliest same-game parlays."""
    r = parlays.across_games(upcoming, 4, cap)
    if not r:
        return ""
    cum = np.cumprod([lg["p"] for lg in r["legs"]])
    across = "".join(f'<li>{leg_text(lg, game=True)}{f" → <b>first {i + 1}: {pc(c)}</b>, fair {american(c)}" if i else ""}</li>'
                     for i, (lg, c) in enumerate(zip(r["legs"], cum)))
    sgps = sorted(((g, g["parlays"][f"3|{cap}"]) for g in upcoming if f"3|{cap}" in g.get("parlays", {})),
                  key=lambda x: -x[1]["p"])[:5]
    same = "".join(f'<li><b>{g["away"]} at {g["home"]}: {pc(p["p"])} together</b>, fair {american(p["p"])} '
                   f'(multiplying the legs {pc(p["ind"])}): '
                   + " + ".join(f'{html.escape(lg["n"])} {lg["label"]}' for lg in p["legs"]) + '</li>' for g, p in sgps)
    return (f'<div class="slate"><h3>Most likely parlays</h3>'
            f'<p class="tdcap">Built from “X+” milestone and touchdown bets, every leg at most {cap:.0%} likely on its '
            f'own (uncapped, the likeliest parlay just stacks near-certain legs that pay nothing). Same-game chances '
            f'count how often the legs hit together in the simulated games, one leg per player. Other leg counts, '
            f'caps and same-player stacks: the Parlays tab of the page. Backtest (406 games, 2025–26): 3–5-leg '
            f'suggestions hit within ~2 points of their predicted chance at these caps; 2-leg ones ~4 points less '
            f'often.</p>'
            f'<p class="tdcap"><b>Across games</b> (one leg per game; games are independent):</p><ol class="plist">{across}</ol>'
            f'<p class="tdcap"><b>Likeliest 3-leg same-game parlays</b> (each game’s best is on its page):</p>'
            f'<ul class="plist">{same}</ul></div>')


def game_section(g):
    qb_new = {p["t"] for p in g["players"] if p.get("new") and p["p"] == "QB"}
    qbs = " vs ".join(f'{html.escape(str(g[s + "_qb"]))}{" (new starter)" if g[s] in qb_new else ""}'
                      for s in ("away", "home"))
    implied = (f' · implied {g["away"]} {g["away_tt"]:g}, {g["home"]} {g["home_tt"]:g}'
               if g.get("away_tt") is not None else "")
    total = f' · total {g["total"]:g}' if g.get("total") is not None else ""
    venue = html.escape(g.get("stadium") or "") + (" (neutral site)" if g.get("neutral") else "")
    f = g.get("final")
    if f:
        if g.get("away_tt") is not None:
            implied = (f' · implied {g["away"]} {g["away_tt"]:g} (scored {f["away"]}), '
                       f'{g["home"]} {g["home_tt"]:g} (scored {f["home"]})')
        if g.get("total") is not None:
            total = f' · total {g["total"]:g} (actual {f["away"] + f["home"]})'
        d = dt.date.fromisoformat(g["day"])
        title = (f'<span>{g["away"]} {f["away"]}<small>at</small>{g["home"]} {f["home"]}</span>'
                 f'<span class="when final">Final · {d.strftime("%a %b")} {d.day}</span>')
        pre = "Pre-game: "
        rows = graded([g])
        card = f'<p class="score">How the projections did: {scorecard(rows)}</p>' if rows else ""
    else:
        title = (f'<span>{g["away"]}<small>at</small>{g["home"]}</span>'
                 f'<span class="when">{kickoff(g, with_date=True)}</span>')
        pre, card = "", ""
    head = (f'<div class="gh"><h2>{title}</h2>'
            f'<div class="meta">{pre}{spread_text(g)}{total}{implied} · {venue}</div>'
            f'<div class="meta">QBs: {qbs}</div>{parlay_line(g)}{card}</div>')
    tables = [market_table(g, k, label) for k, label in MARKETS] + [td_table(g)]
    tables = [t for t in tables if t]
    first = f'<div class="keep">{head}<div class="mk">{tables[0]}</div></div>' if tables else head
    rest = "".join(f'<div class="mk">{t}</div>' for t in tables[1:])
    return f'<section class="game">{first}{rest}</section>'


def legend_svg():
    w, h = 240, 34
    a, c, m, d, b = 20, 70, 110, 160, 220  # 10th, 25th, median, 75th, 90th
    lab = [(a, "10th"), (c, "25th"), (m, "median"), (d, "75th"), (b, "90th")]
    t = "".join(f'<text x="{x}" y="31" text-anchor="middle">{s}</text>' for x, s in lab)
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect class="b80" x="{a}" y="6" width="{b - a}" height="9"/>'
            f'<rect class="b50" x="{c}" y="6" width="{d - c}" height="9"/>'
            f'<rect class="md" x="{m - 1}" y="2" width="2" height="17"/>{t}</svg>')


def slate_row(g):
    total = "" if g.get("total") is None else f'{g["total"]:g}'
    implied = "" if g.get("away_tt") is None else f'{g["away_tt"]:g} – {g["home_tt"]:g}'
    f = g.get("final")
    when = "Final" if f else "Started" if g.get("started") else kickoff(g)
    game = (f'{g["away"]} {f["away"]} at {g["home"]} {f["home"]}' if f else f'{g["away"]} at {g["home"]}')
    return (f'<tr><td>{when}</td><td><b>{game}</b></td><td>{spread_text(g, short=True)}</td>'
            f'<td class="num">{total}</td><td class="num">{implied}</td>'
            f'<td>{html.escape(str(g["away_qb"]))} vs {html.escape(str(g["home_qb"]))}</td></tr>')


def results_note(games):
    """Cover-page summary of every final game so far."""
    rows = graded(games)
    if not rows:
        return ""
    finals = "; ".join(f'{g["away"]} {g["final"]["away"]}, {g["home"]} {g["final"]["home"]}'
                       for g in games if g.get("final"))
    misses = sorted((r for r in rows if r["pct"] < 10 or r["pct"] > 90), key=lambda r: abs(r["pct"] - 50),
                    reverse=True)
    miss_txt = ", ".join(f'{html.escape(r["p"]["n"])} {dict(MARKETS)[r["k"]].lower()} {fmt(r["y"], r["k"])} '
                         f'(median {fmt(r["q"][IDX[0.5]], r["k"])}, {ordinal(r["pct"])} percentile)'
                         for r in misses[:4])
    return (f'<div class="results"><h3>Results so far</h3><p><b>Final:</b> {finals}. {scorecard(rows)} '
            f'(One game is a small sample: game script moves all its props together.)</p>'
            + (f'<p><b>Outside the middle 80%:</b> {miss_txt}.</p>' if misses else "") + '</div>')


def methods_note(games):
    """Cover-page note on the two methods, with the weights and backtest numbers from blend.json."""
    if not any(p.get("s") for g in games for p in g["players"]) or not os.path.exists("blend.json"):
        return ""
    b = json.load(open("blend.json"))
    nm = {"passing_yards": "passing", "rushing_yards": "rushing", "receiving_yards": "receiving yards",
          "receptions": "receptions"}
    wts = ", ".join(f'{nm[m]} {b["blend"][m]["sim_weight"]:.0%}' for m in nm if m in b["blend"])
    gains = ", ".join(f'{nm[m]} {b["report"][m]["season_level_check_pct"]:+.1f}%' for m in nm if m in b["report"])
    return (f'<p><b>Two methods, blended.</b> Each median mixes the stats model with a game simulation (every game '
            f'played snap by snap 1,000 times with each team’s play-calling, tempo and 4th-down habits and each '
            f'player’s usage, tuned to the betting market). Simulation weight: {wts}. On unseen games the blend beat '
            f'the stats model alone: {gains}. “Stats · Sim” shows both medians; amber = they differ by 20%+.</p>'
            + td_note(b))


def td_note(b):
    t = b.get("td")
    if not t:
        return ""
    r = t.get("report", {})
    g = r.get("test_gain_pct")
    gain = (f' On {t["chosen_on"][-1]}’s games (not used to choose the weight) the blend’s log loss was '
            f'{abs(g):.1f}% {"lower (better)" if g >= 0 else "higher (worse)"} than the stats model’s alone.'
            if g is not None else "")
    return (f'<p><b>Touchdowns.</b> Chance of at least one rushing or receiving TD (no passing or return TDs), '
            f'blending the stats model (TD history, red-zone and goal-line role, implied points, opponent) with the '
            f'simulation ({t["sim_weight"]:.0%} weight).{gain} “Fair” = break-even odds; First TD = the game’s first '
            f'touchdown, from the simulation. Books take a big margin on touchdown markets: look for prices well above '
            f'fair. Amber = the methods differ by 8+ points.</p>')


def top_td(upcoming, n=20):
    ps = sorted(((p, g) for g in upcoming for p in g["players"] if p.get("td")), key=lambda x: -x[0]["td"]["m"][0])[:n]
    if not ps:
        return ""
    items = "".join(f'<li><span><b>{html.escape(p["n"])}</b><span class="who">{p["t"]} {p["p"]} vs '
                    f'{g["home"] if p["t"] == g["away"] else g["away"]}</span></span>'
                    f'<span>{pc(p["td"]["m"][0])} <span class="who">fair {american(p["td"]["m"][0])}</span></span></li>'
                    for p, g in ps)
    return (f'<div class="slate"><h3>Likeliest touchdown scorers this week</h3><ul class="toptd">{items}</ul>'
            f'<p class="tdcap">Anytime touchdown chance and break-even (“fair”) odds; each game’s full list is on its '
            f'page.</p></div>')


def build_html(games, season, week, run_time):
    upcoming = [g for g in games if not g.get("started")]
    n_props = sum(len(p["m"]) for g in upcoming for p in g["players"])
    n_final = sum(1 for g in games if g.get("final"))
    slate = "".join(slate_row(g) for g in games)
    new_qbs = [f'{html.escape(p["n"])} ({p["t"]})' for g in upcoming for p in g["players"] if p.get("new")]
    new_qb_note = (f'<p><b>New starting QBs:</b> {", ".join(new_qbs)}. Confirm them in the news (the schedule data has '
                   f'listed wrong starters before) and treat their passing as optimistic: in 2025–26 both methods ran '
                   f'~20 yards high on new starters (backtest through 2026 week 8; not the case in 2021–23).</p>'
                   if new_qbs else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Week {week} prop projections</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<h1>Week {week} prop projections</h1>
<p class="sub">{season} NFL season · model run {run_time} · {len(upcoming)} upcoming games · {n_props} props{f" · {n_final} final" if n_final else ""}</p>
<div class="howto">
  <div>{legend_svg()}</div>
  <div>
    <h3>How to read it</h3>
    <p>The <b>median</b> is the model’s 50/50 line. Dark bar: the middle 50% of likely outcomes (25th–75th
    percentile). Light bar: the middle 80% (10th–90th).</p>
    <p>Against a sportsbook line: at the median it’s a coin flip; at the left end of the dark bar the over hits
    about 75% of the time, at the left end of the light bar about 90% (right ends: the same for the under).
    At −110 you need 52.4% to break even. Bars use one scale per stat (passing 50–400 yds, rushing and receiving
    0–160, receptions 0–12). Exact probabilities for any line: week{week}_projections.html.</p>
  </div>
</div>
<div class="slate"><h3>The slate</h3>
<table><thead><tr><th>Kickoff</th><th>Game</th><th>Favorite</th><th class="num">Total</th>
<th class="num">Implied pts</th><th>Starting QBs</th></tr></thead><tbody>{slate}</tbody></table></div>
{results_note(games)}
<div class="page2">{top_td(upcoming)}{parlay_section(upcoming)}
<div class="notes"><h3>Before you bet</h3>{methods_note(games)}{new_qb_note}
<p><b>Flags</b> = latest injury report; Returning = missed recent games; New team = changed teams. Projections assume
flagged players play: rerun after Friday’s final designations and check the inactive lists (out about 90 minutes before
kickoff): when a player sits, his teammates’ numbers, touchdowns included, go up. Estimates, not guarantees.</p></div></div>
<div class="games-start">{"".join(game_section(g) for g in games)}</div>
</body></html>"""


def find_chrome():
    cands = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
             "/Applications/Chromium.app/Contents/MacOS/Chromium",
             "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe",
             r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    cands += [shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
                                          "chrome", "msedge")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise SystemExit("make_pdf.py prints with Google Chrome (or Chromium/Edge), and none was found.")


def print_pdf(page_html, pdf_path):
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp, "page.html")
        src.write_text(page_html, encoding="utf-8")
        cmd = [find_chrome(), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               f"--user-data-dir={pathlib.Path(tmp, 'profile')}", "--no-pdf-header-footer",
               "--print-to-pdf-no-header", "--virtual-time-budget=8000", f"--print-to-pdf={pdf_path}", src.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0 or not os.path.exists(pdf_path):
            raise SystemExit(f"Chrome could not print the PDF:\n{r.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int)
    args = ap.parse_args()
    src = latest_week(args.week)
    games = json.load(open(src))
    if not games:
        raise SystemExit(f"{src} has no games.")
    week = int(re.search(r"week(\d+)\.json$", src).group(1))
    season = int(games[0]["id"][:4])
    stamp = dt.datetime.fromtimestamp(os.path.getmtime(src))
    run_time = stamp.strftime("%A, %b %-d at %-I:%M %p") if os.name != "nt" else stamp.strftime("%A, %b %d")
    pdf = os.path.abspath(f"output/week{week}_projections.pdf")
    if os.path.exists(pdf):
        os.remove(pdf)
    print_pdf(build_html(games, season, week, run_time), pdf)
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
