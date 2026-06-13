"""Prediction freezing + self-scoring (the public "receipts").

Predictions for fixtures kicking off within 72h are appended (once) to an
append-only log that the daily GitHub Action commits — a timestamped, auditable
record that the forecast was made BEFORE kickoff. As results arrive we grade the
frozen predictions (Brier, log loss, favourite-correct, exact-score-correct).
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config as C

LOG_PATH = C.FROZEN / "predictions_log.csv"
SCORES_PATH = C.FROZEN / "scores.csv"

LOG_FIELDS = ["frozen_at_utc", "match_id", "kickoff_utc", "stage", "home", "away",
              "p_home", "p_draw", "p_away", "pred_score_home", "pred_score_away",
              "model_version"]
SCORE_FIELDS = ["match_id", "kickoff_utc", "home", "away", "outcome",
                "p_home", "p_draw", "p_away", "brier", "logloss",
                "favorite_correct", "exact_correct", "pred_score_home",
                "pred_score_away", "actual_home", "actual_away", "scored_at_utc"]

LABELS = ["home_win", "draw", "away_win"]


def _read_csv(path, fields):
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def freeze_due(fixtures: pd.DataFrame, predictions: dict, model_version: str,
               now: datetime | None = None) -> int:
    """Append predictions for fixtures kicking off within 72h and not yet frozen.

    predictions: {match_id: {"p_home","p_draw","p_away","score_home","score_away"}}
    Returns the number of newly frozen rows. Append-only: existing rows untouched.
    """
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(hours=72)
    existing = {int(r["match_id"]) for r in _read_csv(LOG_PATH, LOG_FIELDS)}
    new_rows = []
    for _, m in fixtures.iterrows():
        mid = int(m["match_id"])
        if mid in existing or mid not in predictions:
            continue
        if m["home_team"] is None or m["away_team"] is None:
            continue
        kickoff = datetime.fromisoformat(m["utc_date"].replace("Z", "+00:00"))
        if not (now <= kickoff <= horizon):
            continue
        p = predictions[mid]
        new_rows.append({
            "frozen_at_utc": now.isoformat(),
            "match_id": mid,
            "kickoff_utc": m["utc_date"],
            "stage": m["stage"],
            "home": m["home_team"],
            "away": m["away_team"],
            "p_home": round(p["p_home"], 4),
            "p_draw": round(p["p_draw"], 4),
            "p_away": round(p["p_away"], 4),
            "pred_score_home": p["score_home"],
            "pred_score_away": p["score_away"],
            "model_version": model_version,
        })
    if new_rows:
        header = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if header:
                w.writeheader()
            w.writerows(new_rows)
    return len(new_rows)


def _brier(probs, outcome_idx):
    onehot = np.zeros(3)
    onehot[outcome_idx] = 1.0
    return float(np.sum((np.array(probs) - onehot) ** 2))


def score_resolved(fixtures: pd.DataFrame, now: datetime | None = None) -> int:
    """Grade frozen predictions whose results are now known. Returns count scored."""
    now = now or datetime.now(timezone.utc)
    frozen = _read_csv(LOG_PATH, LOG_FIELDS)
    if not frozen:
        return 0
    already = {int(r["match_id"]) for r in _read_csv(SCORES_PATH, SCORE_FIELDS)}
    results = {int(m["match_id"]): m for _, m in fixtures.iterrows()
               if m["played"] and m["home_score"] is not None}

    scored = _read_csv(SCORES_PATH, SCORE_FIELDS)
    n_new = 0
    for r in frozen:
        mid = int(r["match_id"])
        if mid in already or mid not in results:
            continue
        m = results[mid]
        ah, aa = int(m["home_score"]), int(m["away_score"])
        if ah > aa:
            outcome, oidx = "home_win", 0
        elif ah < aa:
            outcome, oidx = "away_win", 2
        else:
            outcome, oidx = "draw", 1
        probs = [float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])]
        fav = int(np.argmax(probs))
        scored.append({
            "match_id": mid, "kickoff_utc": r["kickoff_utc"],
            "home": r["home"], "away": r["away"], "outcome": outcome,
            "p_home": r["p_home"], "p_draw": r["p_draw"], "p_away": r["p_away"],
            "brier": round(_brier(probs, oidx), 4),
            "logloss": round(float(-np.log(max(probs[oidx], 1e-12))), 4),
            "favorite_correct": int(fav == oidx),
            "exact_correct": int(int(r["pred_score_home"]) == ah and int(r["pred_score_away"]) == aa),
            "pred_score_home": r["pred_score_home"], "pred_score_away": r["pred_score_away"],
            "actual_home": ah, "actual_away": aa,
            "scored_at_utc": now.isoformat(),
        })
        n_new += 1
    if n_new:
        _write_csv(SCORES_PATH, SCORE_FIELDS, scored)
    return n_new
