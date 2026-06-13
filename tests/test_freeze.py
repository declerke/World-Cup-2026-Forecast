"""Prediction freezing: append-only behaviour and scoring math."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import freeze
import config as C


@pytest.fixture
def tmp_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "LOG_PATH", tmp_path / "log.csv")
    monkeypatch.setattr(freeze, "SCORES_PATH", tmp_path / "scores.csv")
    return tmp_path


def _fixture_row(mid, kickoff, home="A", away="B", played=False, hs=None, as_=None):
    return {
        "match_id": mid, "utc_date": kickoff, "stage": "group",
        "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_, "played": played,
    }


def test_freeze_only_within_72h(tmp_frozen):
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    soon = (now + timedelta(hours=24)).isoformat().replace("+00:00", "Z")
    far = (now + timedelta(days=10)).isoformat().replace("+00:00", "Z")
    fx = pd.DataFrame([_fixture_row(1, soon), _fixture_row(2, far)])
    preds = {1: {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "score_home": 1, "score_away": 0},
             2: {"p_home": 0.4, "p_draw": 0.3, "p_away": 0.3, "score_home": 1, "score_away": 1}}
    n = freeze.freeze_due(fx, preds, "testsha", now=now)
    assert n == 1   # only the within-72h match


def test_freeze_is_append_only(tmp_frozen):
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    soon = (now + timedelta(hours=24)).isoformat().replace("+00:00", "Z")
    fx = pd.DataFrame([_fixture_row(1, soon)])
    preds = {1: {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "score_home": 1, "score_away": 0}}
    assert freeze.freeze_due(fx, preds, "v1", now=now) == 1
    # second call must NOT re-freeze the same match
    assert freeze.freeze_due(fx, preds, "v2", now=now) == 0
    rows = freeze._read_csv(freeze.LOG_PATH, freeze.LOG_FIELDS)
    assert len(rows) == 1
    assert rows[0]["model_version"] == "v1"   # original preserved


def test_scoring_math(tmp_frozen):
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    soon = (now + timedelta(hours=24)).isoformat().replace("+00:00", "Z")
    fx = pd.DataFrame([_fixture_row(1, soon)])
    preds = {1: {"p_home": 0.6, "p_draw": 0.25, "p_away": 0.15, "score_home": 2, "score_away": 0}}
    freeze.freeze_due(fx, preds, "v1", now=now)
    # resolve: home win 2-0 (favorite + exact correct)
    fx_done = pd.DataFrame([_fixture_row(1, soon, played=True, hs=2, as_=0)])
    assert freeze.score_resolved(fx_done, now=now) == 1
    s = freeze._read_csv(freeze.SCORES_PATH, freeze.SCORE_FIELDS)[0]
    assert s["outcome"] == "home_win"
    assert int(s["favorite_correct"]) == 1
    assert int(s["exact_correct"]) == 1
    # Brier = (0.6-1)^2 + (0.25-0)^2 + (0.15-0)^2 = 0.16+0.0625+0.0225 = 0.245
    assert float(s["brier"]) == pytest.approx(0.245, abs=1e-3)


def test_scoring_wrong_favorite(tmp_frozen):
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    soon = (now + timedelta(hours=24)).isoformat().replace("+00:00", "Z")
    fx = pd.DataFrame([_fixture_row(1, soon)])
    preds = {1: {"p_home": 0.6, "p_draw": 0.25, "p_away": 0.15, "score_home": 2, "score_away": 0}}
    freeze.freeze_due(fx, preds, "v1", now=now)
    fx_done = pd.DataFrame([_fixture_row(1, soon, played=True, hs=0, as_=1)])  # away win
    freeze.score_resolved(fx_done, now=now)
    s = freeze._read_csv(freeze.SCORES_PATH, freeze.SCORE_FIELDS)[0]
    assert int(s["favorite_correct"]) == 0
    assert int(s["exact_correct"]) == 0
