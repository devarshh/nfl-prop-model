"""How the game simulator relates to the main model, and how much weight each should get.

    python blend.py gbm_backtest.parquet [more.parquet ...]

Inputs: sim_backtest.parquet (from backtest_sim.py) and the main model's walk-forward backtest for the same
games (backtest_all.parquet from backtest.py, one file per season is fine).

Blending averages the two models' quantiles: (1 - weight) * model + weight * scale * sim. Averaging quantiles
keeps the result as sharp as its parts; averaging probabilities would make every prop look less certain than
either model says. The weight is stable across seasons, but the simulator's overall level drifts from season to
season (2025: best scaled down ~6%; 2026: up ~4%), so the scale is fitted on the current season's finished
games. Refresh it every few weeks: python backtest_sim.py 2025 2026 && python backtest.py && python blend.py ...

Writes blend.json for run.py: per stat, the simulator's weight (all seasons) and scale (latest season).
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from model import QUANTILES, qcol

LV = np.array(QUANTILES)
TAUS = np.arange(0.005, 1.0, 0.01)
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)
SCALES = np.round(np.arange(0.90, 1.061, 0.02), 2)  # correction for any steady lean in the simulator's numbers


def quantile_fn(Q, taus=TAUS):
    """Quantile function at taus from 7 predicted quantiles: straight lines between them and the same
    exponential tails as model.cdf."""
    Q = np.sort(Q, axis=1)
    out = np.empty((len(Q), len(taus)))
    inner = (taus >= LV[0]) & (taus <= LV[-1])
    for j in np.flatnonzero(inner):
        k = min(max(np.searchsorted(LV, taus[j]), 1), len(LV) - 1)
        w = (taus[j] - LV[k - 1]) / (LV[k] - LV[k - 1])
        out[:, j] = Q[:, k - 1] + w * (Q[:, k] - Q[:, k - 1])
    lo, hi = taus < LV[0], taus > LV[-1]
    b_lo = LV[0] * np.maximum(Q[:, 1] - Q[:, 0], 1e-6) / (LV[1] - LV[0])
    b_hi = (1 - LV[-1]) * np.maximum(Q[:, -1] - Q[:, -2], 1e-6) / (LV[-1] - LV[-2])
    out[:, lo] = Q[:, [0]] + b_lo[:, None] * np.log(taus[lo] / LV[0])
    out[:, hi] = Q[:, [-1]] - b_hi[:, None] * np.log((1 - taus[hi]) / (1 - LV[-1]))
    return out


def crps(Q, y):
    """CRPS = 2 x average pinball loss over all quantile levels (lower is better)."""
    q = quantile_fn(Q)
    d = y[:, None] - q
    return 2 * np.mean(np.maximum(TAUS * d, (TAUS - 1) * d), axis=1)


def load_gbm(paths):
    d = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    ren = {}
    for c in d.columns:  # accept q0.05-style (scratch backtests) and q5-style (backtest.py) names
        m = re.fullmatch(r"q(0\.\d+)", c)
        if m:
            ren[c] = qcol(float(m.group(1)))
    return d.rename(columns=ren)


def main(paths):
    gbm = load_gbm(paths)
    sim = pd.read_parquet("sim_backtest.parquet")
    qc = [qcol(q) for q in QUANTILES]
    m = gbm.merge(sim, on=["game_id", "player_id", "market"], suffixes=("_g", "_s"))
    m["season"] = m.game_id.str[:4].astype(int)
    seasons = sorted(m.season.unique())
    fit, test = seasons[:-1] or seasons, seasons[-1:]
    rng = np.random.default_rng(0)
    report, weights = {}, {}
    print(f"{len(m)} props matched ({m.game_id.nunique()} games); weights chosen on {fit}, checked on {test}\n")
    for mk, g in m.groupby("market"):
        y = g.actual.to_numpy(float)
        if mk == "receptions":  # both models describe receptions + uniform(-0.5, 0.5)
            y = y + rng.uniform(-0.5, 0.5, len(y))
        Qg, Qs = g[[c + "_g" for c in qc]].to_numpy(float), g[[c + "_s" for c in qc]].to_numpy(float)
        f, t = g.season.isin(fit).to_numpy(), g.season.isin(test).to_numpy()
        scores = {(c, w): crps(w * Qg + (1 - w) * c * Qs, y) for c in SCALES for w in WEIGHTS}
        c_fit, w_fit = min(scores, key=lambda cw: scores[cw][f].mean())
        c_all, w_all = min(scores, key=lambda cw: scores[cw].mean())
        c_sim = min(SCALES, key=lambda c: scores[(c, 0.0)][f].mean())  # best the simulator does on its own
        mid = QUANTILES.index(0.5)
        med_g, med_s = Qg[:, mid], Qs[:, mid]
        r = {
            "n": int(len(g)), "corr_medians": float(np.corrcoef(med_g, med_s)[0, 1]),
            "bias_model": float(np.mean(med_g - g.actual)), "bias_sim": float(np.mean(med_s - g.actual)),
            "MAE_model": float(np.mean(np.abs(med_g - g.actual))), "MAE_sim": float(np.mean(np.abs(med_s - g.actual))),
            "CRPS_model_test": float(scores[(1.0, 1.0)][t].mean()), "CRPS_sim_test": float(scores[(c_sim, 0.0)][t].mean()),
            "CRPS_blend_test": float(scores[(c_fit, w_fit)][t].mean()), "model_weight_fit": float(w_fit),
            "sim_scale_fit": float(c_fit), "model_weight_all": float(w_all), "sim_scale_all": float(c_all),
            "sim_coverage": {qcol(q): float(np.mean(g.actual.to_numpy() <= Qs[:, i])) for i, q in enumerate(QUANTILES)},
        }
        w_fit = (c_fit, w_fit)
        scores[1.0] = scores[(1.0, 1.0)]
        # is the blend's gain on the test season real? paired difference, clustered by game
        diff = pd.Series(scores[w_fit][t] - scores[1.0][t]).groupby(g.game_id.to_numpy()[t]).sum()
        n_t = t.sum()
        boot = [diff.sample(frac=1, replace=True, random_state=i).sum() / n_t for i in range(500)]
        r["test_gain_pct"] = float(-diff.sum() / n_t / scores[1.0][t].mean() * 100)
        r["test_gain_se_pct"] = float(np.std(boot) / scores[1.0][t].mean() * 100)
        # production: weight from all seasons, level from the latest one. Its honest check: weight from earlier
        # seasons + level from the first half of the latest season, scored on that season's second half
        last = g.season.eq(seasons[-1]).to_numpy()
        wk = g.game_id.str[5:7].astype(int).to_numpy()
        split = np.median(wk[last])
        early, late = last & (wk <= split), last & (wk > split)
        w_prod = 1.0 - min(WEIGHTS, key=lambda w: min(scores[(c, w)].mean() for c in SCALES))
        c_prod = min(SCALES, key=lambda c: scores[(c, 1 - w_prod)][last].mean())
        c_early = min(SCALES, key=lambda c: scores[(c, w_fit[1])][early].mean())
        r["season_level_check_pct"] = float(-(scores[(c_early, w_fit[1])][late].mean() / scores[1.0][late].mean() - 1) * 100)
        r["sim_weight"], r["sim_scale"] = float(round(w_prod, 2)), float(c_prod)
        report[mk] = r
        weights[mk] = {"sim_weight": round(float(w_prod), 2), "sim_scale": float(c_prod)}
        print(f"{mk}: n={r['n']}, medians correlate {r['corr_medians']:.2f}; median bias model {r['bias_model']:+.2f}, "
              f"sim {r['bias_sim']:+.2f}; MAE model {r['MAE_model']:.2f}, sim {r['MAE_sim']:.2f}")
        print(f"   test-season CRPS: model {r['CRPS_model_test']:.4f}, sim {r['CRPS_sim_test']:.4f} (scale {c_sim}), "
              f"blend {r['CRPS_blend_test']:.4f} (model weight {w_fit[1]:.1f}, sim scale {w_fit[0]}) -> "
              f"{r['test_gain_pct']:+.2f}% ± {r['test_gain_se_pct']:.2f}% vs model alone")
        print(f"   with this season's level (fit on its first half, scored on its second half): "
              f"{r['season_level_check_pct']:+.2f}% vs model alone")
        print("   sim coverage:", {k: round(v, 3) for k, v in r["sim_coverage"].items()})
    out = json.load(open("blend.json")) if os.path.exists("blend.json") else {}  # keep touchdowns.py's "td" entry
    out.update({"blend": weights, "chosen_on": [int(s) for s in seasons], "report": report})
    json.dump(out, open("blend.json", "w"), indent=1)
    print("\nwrote blend.json:", weights)


if __name__ == "__main__":
    import glob
    main(sys.argv[1:] or sorted(glob.glob("backtest_20[0-9][0-9].parquet")))
