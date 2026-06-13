"""World Football Elo engine.

Standard Elo (eloratings.net formulation) computed from the full 1872+ history
in a single chronological pass. For every match we record each side's *pre-match*
rating, which is exactly what leakage-safe features need. We also keep the full
per-team rating timeline for trend features and the frontend Elo charts.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

import config as C


def _expected(elo_a: float, elo_b: float, home_adv: float) -> float:
    """Expected score for A given a home-advantage shift (0 on neutral ground)."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b + home_adv) / 400.0))


def _gd_multiplier(goal_diff: int) -> float:
    """Goal-difference weighting from the World Football Elo system."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def run_elo(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Compute Elo over the full history.

    Returns
    -------
    matches : history + columns [elo_home_pre, elo_away_pre]
    final   : {team: latest Elo}
    timeline: long frame [date, team, elo] of post-match ratings (for trends/charts)
    """
    history = history.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = defaultdict(lambda: C.ELO_START)

    elo_home_pre = np.empty(len(history))
    elo_away_pre = np.empty(len(history))
    tl_dates, tl_teams, tl_elos = [], [], []

    homes = history["home_team"].to_numpy()
    aways = history["away_team"].to_numpy()
    hs = history["home_score"].to_numpy()
    as_ = history["away_score"].to_numpy()
    neutral = history["neutral"].to_numpy()
    buckets = history["bucket"].to_numpy()
    dates = history["date"].to_numpy()

    for i in range(len(history)):
        h, a = homes[i], aways[i]
        rh, ra = ratings[h], ratings[a]
        elo_home_pre[i] = rh
        elo_away_pre[i] = ra

        home_adv = 0.0 if neutral[i] else C.ELO_HOME_ADVANTAGE
        we_home = _expected(rh, ra, home_adv)

        if hs[i] > as_[i]:
            w_home = 1.0
        elif hs[i] < as_[i]:
            w_home = 0.0
        else:
            w_home = 0.5

        k = C.ELO_K[buckets[i]]
        g = _gd_multiplier(hs[i] - as_[i])
        delta = k * g * (w_home - we_home)

        ratings[h] = rh + delta
        ratings[a] = ra - delta            # zero-sum

        tl_dates.extend([dates[i], dates[i]])
        tl_teams.extend([h, a])
        tl_elos.extend([ratings[h], ratings[a]])

    matches = history.copy()
    matches["elo_home_pre"] = elo_home_pre
    matches["elo_away_pre"] = elo_away_pre

    timeline = pd.DataFrame({"date": tl_dates, "team": tl_teams, "elo": tl_elos})
    final = dict(ratings)
    return matches, final, timeline


def win_probability(elo_a: float, elo_b: float, *, neutral: bool = True) -> float:
    """Pre-match expected score for A (used by the simulator for KO tiebreaks)."""
    home_adv = 0.0 if neutral else C.ELO_HOME_ADVANTAGE
    return _expected(elo_a, elo_b, home_adv)


if __name__ == "__main__":
    import ingest
    matches, final, timeline = run_elo(ingest.build_history())
    top = sorted(final.items(), key=lambda kv: -kv[1])[:12]
    print("Top 12 by current Elo:")
    for t, e in top:
        print(f"  {t:<18} {e:7.1f}")
