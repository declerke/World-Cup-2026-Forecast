"""Leakage-safe feature engineering.

Every feature for a given match is computed using ONLY matches dated strictly
before it (walk-forward). We make one chronological pass, snapshot each team's
state *before* applying a match, emit the feature row, then update state. The
same TeamState snapshots are reused to build feature vectors for upcoming
fixtures and for arbitrary knockout pairings during simulation.
"""
from __future__ import annotations

import bisect
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C
import venues

IMPORTANCE_ORDINAL = {
    "friendly": 0,
    "other_tournament": 1,
    "qualifier": 2,
    "continental_final": 3,
    "world_cup": 4,
}

# Canonical feature column order (persisted to models/feature_list.json).
FEATURE_COLUMNS = [
    "elo_home", "elo_away", "elo_diff",
    "neutral", "home_is_host",
    "form5_win_h", "form5_draw_h", "form5_gf_h", "form5_ga_h",
    "form5_win_a", "form5_draw_a", "form5_gf_a", "form5_ga_a",
    "form10_win_h", "form10_draw_h", "form10_gf_h", "form10_ga_h",
    "form10_win_a", "form10_draw_a", "form10_gf_a", "form10_ga_a",
    "form10_oppelo_h", "form10_oppelo_a",
    "rest_h", "rest_a",
    "h2h_home_winrate", "h2h_mean_gd",
    "importance",
    "elo_trend_h", "elo_trend_a",
    "alt_gap_home", "alt_gap_away",
]


@dataclass
class TeamState:
    elo: float = C.ELO_START
    results: deque = field(default_factory=lambda: deque(maxlen=10))   # 1/0.5/0
    gf: deque = field(default_factory=lambda: deque(maxlen=10))
    ga: deque = field(default_factory=lambda: deque(maxlen=10))
    opp_elo: deque = field(default_factory=lambda: deque(maxlen=10))
    last_date: pd.Timestamp | None = None
    elo_dates: list = field(default_factory=list)     # sorted timeline for trend
    elo_hist: list = field(default_factory=list)


def _form(results, gf, ga, n):
    """Win rate, draw rate, gf avg, ga avg over the last n matches (defaults when empty)."""
    r = list(results)[-n:]
    if not r:
        return 0.5, 0.25, 1.0, 1.0
    wins = sum(1 for x in r if x == 1.0) / len(r)
    draws = sum(1 for x in r if x == 0.5) / len(r)
    g = list(gf)[-n:]
    a = list(ga)[-n:]
    return wins, draws, float(np.mean(g)), float(np.mean(a))


def _elo_trend(state: TeamState, ref_date) -> float:
    """Elo change over the trailing 365 days (0 if no history that far back)."""
    if not state.elo_dates:
        return 0.0
    cutoff = ref_date - np.timedelta64(365, "D")
    idx = bisect.bisect_left(state.elo_dates, cutoff)
    if idx >= len(state.elo_hist):
        idx = len(state.elo_hist) - 1
    return state.elo - state.elo_hist[idx]


def _rest_days(last_date, ref_date) -> float:
    if last_date is None:
        return float(C.MAX_REST_DAYS)
    d = (ref_date - last_date) / np.timedelta64(1, "D")
    return float(min(max(d, 0.0), C.MAX_REST_DAYS))


def _gd_multiplier(goal_diff: int) -> float:
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _expected(elo_a, elo_b, home_adv):
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b + home_adv) / 400.0))


def _pair_features(hs: TeamState, as_: TeamState, neutral: bool, home_is_host: bool,
                   importance: int, ref_date, h2h,
                   alt_gap_home: float = 0.0, alt_gap_away: float = 0.0) -> dict:
    f5h = _form(hs.results, hs.gf, hs.ga, 5)
    f5a = _form(as_.results, as_.gf, as_.ga, 5)
    f10h = _form(hs.results, hs.gf, hs.ga, 10)
    f10a = _form(as_.results, as_.gf, as_.ga, 10)
    oppe_h = float(np.mean(hs.opp_elo)) if hs.opp_elo else C.ELO_START
    oppe_a = float(np.mean(as_.opp_elo)) if as_.opp_elo else C.ELO_START
    h2h_wr, h2h_gd = h2h
    return {
        "elo_home": hs.elo, "elo_away": as_.elo, "elo_diff": hs.elo - as_.elo,
        "neutral": int(neutral), "home_is_host": int(home_is_host),
        "form5_win_h": f5h[0], "form5_draw_h": f5h[1], "form5_gf_h": f5h[2], "form5_ga_h": f5h[3],
        "form5_win_a": f5a[0], "form5_draw_a": f5a[1], "form5_gf_a": f5a[2], "form5_ga_a": f5a[3],
        "form10_win_h": f10h[0], "form10_draw_h": f10h[1], "form10_gf_h": f10h[2], "form10_ga_h": f10h[3],
        "form10_win_a": f10a[0], "form10_draw_a": f10a[1], "form10_gf_a": f10a[2], "form10_ga_a": f10a[3],
        "form10_oppelo_h": oppe_h, "form10_oppelo_a": oppe_a,
        "rest_h": _rest_days(hs.last_date, ref_date),
        "rest_a": _rest_days(as_.last_date, ref_date),
        "h2h_home_winrate": h2h_wr, "h2h_mean_gd": h2h_gd,
        "importance": importance,
        "elo_trend_h": _elo_trend(hs, ref_date),
        "elo_trend_a": _elo_trend(as_, ref_date),
        "alt_gap_home": alt_gap_home,
        "alt_gap_away": alt_gap_away,
    }


def _h2h_lookup(h2h_log: dict, home: str, away: str) -> tuple[float, float]:
    """Home-team win rate and mean goal diff over the last 10 meetings (either venue)."""
    key = tuple(sorted((home, away)))
    log = h2h_log.get(key)
    if not log:
        return 0.5, 0.0
    recent = log[-10:]
    # log stores (home_team_in_that_match, home_goals, away_goals)
    wins, gds = [], []
    for ht, hg, ag in recent:
        if ht == home:
            wins.append(1.0 if hg > ag else 0.0 if hg < ag else 0.5)
            gds.append(hg - ag)
        else:
            wins.append(1.0 if ag > hg else 0.0 if ag < hg else 0.5)
            gds.append(ag - hg)
    return float(np.mean(wins)), float(np.mean(gds))


def build(history: pd.DataFrame):
    """Walk-forward pass.

    Returns
    -------
    train : DataFrame with FEATURE_COLUMNS + ['date','outcome'] for matches >= TRAIN_FROM
    states: dict[team] -> TeamState  (final snapshot, for inference/simulation)
    h2h_log: dict for pairwise lookups at inference time
    """
    history = history.sort_values("date").reset_index(drop=True)
    states: dict[str, TeamState] = defaultdict(TeamState)
    h2h_log: dict[tuple, list] = defaultdict(list)

    # Altitude: team baselines (a stable geographic attribute — where a team is
    # based — so computing it from all history is not outcome leakage) + per-match
    # venue altitude. Degrades to no-op when the history has no city column.
    if "city" in history.columns:
        baselines = venues.team_baselines(history)
        venue_alt_arr = history["city"].map(venues.altitude).to_numpy()
    else:
        baselines = {}
        venue_alt_arr = np.zeros(len(history))

    rows = []
    train_from = np.datetime64(C.TRAIN_FROM)

    homes = history["home_team"].to_numpy()
    aways = history["away_team"].to_numpy()
    hs_arr = history["home_score"].to_numpy()
    as_arr = history["away_score"].to_numpy()
    neutral_arr = history["neutral"].to_numpy()
    bucket_arr = history["bucket"].to_numpy()
    date_arr = history["date"].to_numpy()

    for i in range(len(history)):
        h, a = homes[i], aways[i]
        hstate, astate = states[h], states[a]
        ref = date_arr[i]
        importance = IMPORTANCE_ORDINAL[bucket_arr[i]]
        home_is_host = not neutral_arr[i]
        h2h = _h2h_lookup(h2h_log, h, a)

        if ref >= train_from:
            v_alt = venue_alt_arr[i]
            agh = max(0.0, v_alt - baselines.get(h, 0.0))
            aga = max(0.0, v_alt - baselines.get(a, 0.0))
            feat = _pair_features(hstate, astate, bool(neutral_arr[i]), home_is_host,
                                  importance, ref, h2h, agh, aga)
            if hs_arr[i] > as_arr[i]:
                outcome = "home_win"
            elif hs_arr[i] < as_arr[i]:
                outcome = "away_win"
            else:
                outcome = "draw"
            feat["date"] = ref
            feat["outcome"] = outcome
            feat["home_goals"] = int(hs_arr[i])
            feat["away_goals"] = int(as_arr[i])
            rows.append(feat)

        # ---- update Elo (same formula as elo.py) ----
        home_adv = 0.0 if neutral_arr[i] else C.ELO_HOME_ADVANTAGE
        we_home = _expected(hstate.elo, astate.elo, home_adv)
        if hs_arr[i] > as_arr[i]:
            w_home = 1.0
        elif hs_arr[i] < as_arr[i]:
            w_home = 0.0
        else:
            w_home = 0.5
        k = C.ELO_K[bucket_arr[i]]
        g = _gd_multiplier(hs_arr[i] - as_arr[i])
        delta = k * g * (w_home - we_home)
        opp_h_elo, opp_a_elo = astate.elo, hstate.elo
        hstate.elo += delta
        astate.elo -= delta

        # ---- update form / h2h / timelines ----
        hstate.results.append(w_home); astate.results.append(1.0 - w_home)
        hstate.gf.append(int(hs_arr[i])); hstate.ga.append(int(as_arr[i]))
        astate.gf.append(int(as_arr[i])); astate.ga.append(int(hs_arr[i]))
        hstate.opp_elo.append(opp_h_elo); astate.opp_elo.append(opp_a_elo)
        hstate.last_date = astate.last_date = ref
        hstate.elo_dates.append(ref); hstate.elo_hist.append(hstate.elo)
        astate.elo_dates.append(ref); astate.elo_hist.append(astate.elo)
        h2h_log[tuple(sorted((h, a)))].append((h, int(hs_arr[i]), int(as_arr[i])))

    train = pd.DataFrame(rows)
    return train, dict(states), dict(h2h_log), baselines


def features_for_pair(home, away, *, neutral, states, h2h_log,
                      ref_date=None, importance=4, rest_default=4.0,
                      baselines=None, venue_altitude=0.0) -> dict:
    """Feature vector for an upcoming/simulated match from current team state.

    venue_altitude (m) and baselines drive the altitude-gap features; both
    default to a sea-level / no-baseline no-op so callers without venue context
    behave exactly as before.
    """
    ref = np.datetime64(ref_date) if ref_date is not None else np.datetime64(C.TEST_TO)
    hstate = states.get(home) or TeamState()
    astate = states.get(away) or TeamState()
    home_is_host = not neutral
    h2h = _h2h_lookup(h2h_log, home, away)
    baselines = baselines or {}
    agh = max(0.0, venue_altitude - baselines.get(home, 0.0))
    aga = max(0.0, venue_altitude - baselines.get(away, 0.0))
    feat = _pair_features(hstate, astate, neutral, home_is_host, importance, ref, h2h,
                          agh, aga)
    # For neutral simulated knockouts we don't know exact dates; use a tournament rest.
    if ref_date is None:
        feat["rest_h"] = feat["rest_a"] = rest_default
    return feat


if __name__ == "__main__":
    import ingest
    train, states, h2h, baselines = build(ingest.build_history())
    print(f"training rows: {len(train):,}")
    print(train["outcome"].value_counts(normalize=True).round(3).to_dict())
    print(f"date range: {train['date'].min()} .. {train['date'].max()}")
    print(f"feature cols: {len(FEATURE_COLUMNS)}  teams with state: {len(states)}")
