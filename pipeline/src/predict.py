"""Model inference: W/D/L probabilities and Poisson scoreline matrices.

Used by the simulator, the prediction-freezer, and the JSON publisher. For
neutral knockout ties we average both home/away orderings so no residual
home-slot bias leaks into a match played at a neutral venue.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
from scipy.stats import poisson

import config as C
import features as F

HOSTS = {"United States", "Mexico", "Canada"}

_MODELS: dict | None = None


def load_models() -> dict:
    global _MODELS
    if _MODELS is None:
        _MODELS = {
            "wdl": joblib.load(C.MODELS / "wdl_model.joblib"),
            "pois_home": joblib.load(C.MODELS / "poisson_home.joblib"),
            "pois_away": joblib.load(C.MODELS / "poisson_away.joblib"),
        }
    return _MODELS


def _vec(feat: dict) -> np.ndarray:
    return np.array([[feat[c] for c in F.FEATURE_COLUMNS]], dtype=float)


def wdl(home, away, *, neutral, states, h2h, importance=4, ref_date=None,
        baselines=None, venue_altitude=0.0):
    """Return (p_home_win, p_draw, p_away_win) for a directed fixture."""
    m = load_models()
    feat = F.features_for_pair(home, away, neutral=neutral, states=states,
                               h2h_log=h2h, importance=importance, ref_date=ref_date,
                               baselines=baselines, venue_altitude=venue_altitude)
    p = m["wdl"].predict_proba(_vec(feat))[0]   # order: [home_win, draw, away_win]
    return float(p[0]), float(p[1]), float(p[2])


def wdl_neutral(a, b, *, states, h2h, importance=4, baselines=None, venue_altitude=0.0):
    """Symmetric neutral-venue probabilities (a vs b), averaging both orderings.

    Returns (p_a_win, p_draw, p_b_win).
    """
    ph, pd1, pa = wdl(a, b, neutral=True, states=states, h2h=h2h, importance=importance,
                      baselines=baselines, venue_altitude=venue_altitude)
    ph2, pd2, pa2 = wdl(b, a, neutral=True, states=states, h2h=h2h, importance=importance,
                        baselines=baselines, venue_altitude=venue_altitude)
    # second call is from b's perspective: ph2 = P(b win), pa2 = P(a win)
    p_a = (ph + pa2) / 2
    p_b = (pa + ph2) / 2
    p_d = (pd1 + pd2) / 2
    s = p_a + p_d + p_b
    return p_a / s, p_d / s, p_b / s


def goal_lambdas(home, away, *, neutral, states, h2h, importance=4, ref_date=None,
                 baselines=None, venue_altitude=0.0):
    m = load_models()
    feat = F.features_for_pair(home, away, neutral=neutral, states=states,
                               h2h_log=h2h, importance=importance, ref_date=ref_date,
                               baselines=baselines, venue_altitude=venue_altitude)
    x = _vec(feat)
    lh = float(np.clip(m["pois_home"].predict(x)[0], C.LAMBDA_MIN, C.LAMBDA_MAX))
    la = float(np.clip(m["pois_away"].predict(x)[0], C.LAMBDA_MIN, C.LAMBDA_MAX))
    return lh, la


def scoreline_matrix(home, away, *, neutral, states, h2h, importance=4, ref_date=None,
                     wdl_probs=None, baselines=None, venue_altitude=0.0):
    """(MAX_GOALS+1) square joint scoreline matrix from independent Poissons,
    region-renormalised so its W/D/L mass matches the classifier (authoritative)."""
    lh, la = goal_lambdas(home, away, neutral=neutral, states=states, h2h=h2h,
                          importance=importance, ref_date=ref_date,
                          baselines=baselines, venue_altitude=venue_altitude)
    n = C.MAX_GOALS + 1
    ph = poisson.pmf(np.arange(n), lh)
    pa = poisson.pmf(np.arange(n), la)
    mat = np.outer(ph, pa)
    mat /= mat.sum()

    if wdl_probs is None:
        wdl_probs = wdl(home, away, neutral=neutral, states=states, h2h=h2h,
                        importance=importance, ref_date=ref_date,
                        baselines=baselines, venue_altitude=venue_altitude)
    p_home, p_draw, p_away = wdl_probs
    idx = np.arange(n)
    home_mask = idx[:, None] > idx[None, :]
    draw_mask = idx[:, None] == idx[None, :]
    away_mask = idx[:, None] < idx[None, :]
    for mask, target in ((home_mask, p_home), (draw_mask, p_draw), (away_mask, p_away)):
        cur = mat[mask].sum()
        if cur > 0:
            mat[mask] *= target / cur
    mat /= mat.sum()
    return mat


def top_scorelines(mat: np.ndarray, k: int = 5):
    """Top-k (home_goals, away_goals, prob) cells of a scoreline matrix."""
    flat = np.argsort(mat, axis=None)[::-1][:k]
    out = []
    for f in flat:
        i, j = divmod(int(f), mat.shape[1])
        out.append({"home": i, "away": j, "prob": round(float(mat[i, j]), 4)})
    return out


def is_neutral(home_team: str) -> bool:
    """A WC 2026 match is non-neutral only when a host nation plays at home."""
    return home_team not in HOSTS
