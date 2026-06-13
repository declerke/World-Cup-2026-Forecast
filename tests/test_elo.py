"""Elo engine: hand-computed cases, branches, zero-sum, convergence."""
import numpy as np
import pandas as pd
import pytest

import elo
import config as C


def test_expected_symmetry():
    # Equal ratings on neutral ground -> 0.5 each
    assert elo._expected(1500, 1500, 0.0) == pytest.approx(0.5)
    # Home advantage shifts expectation above 0.5
    assert elo._expected(1500, 1500, 100.0) > 0.5


def test_expected_known_value():
    # 400-point gap -> 10:1 expected (~0.909)
    assert elo._expected(1900, 1500, 0.0) == pytest.approx(10 / 11, abs=1e-6)


def test_gd_multiplier_branches():
    assert elo._gd_multiplier(0) == 1.0
    assert elo._gd_multiplier(1) == 1.0
    assert elo._gd_multiplier(2) == 1.5
    assert elo._gd_multiplier(3) == (11 + 3) / 8
    assert elo._gd_multiplier(5) == (11 + 5) / 8


def _match(home, away, hs, as_, neutral=True, bucket="friendly", date="2000-01-01"):
    return pd.DataFrame([{
        "date": pd.Timestamp(date), "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_, "neutral": neutral,
        "bucket": bucket, "tournament": "Friendly", "city": None, "country": None,
    }])


def test_single_match_update_zero_sum():
    df = _match("A", "B", 1, 0)
    matches, final, _ = elo.run_elo(df)
    # Expected was 0.5 each (neutral, equal). K=20 friendly, GD=1 -> mult 1.
    # delta = 20 * 1 * (1 - 0.5) = 10
    assert final["A"] == pytest.approx(1510.0)
    assert final["B"] == pytest.approx(1490.0)
    # zero-sum
    assert final["A"] + final["B"] == pytest.approx(2 * C.ELO_START)


def test_pre_match_ratings_recorded():
    df = _match("A", "B", 2, 1)
    matches, _, _ = elo.run_elo(df)
    assert matches.iloc[0]["elo_home_pre"] == C.ELO_START
    assert matches.iloc[0]["elo_away_pre"] == C.ELO_START


def test_world_cup_k_larger_than_friendly():
    wc = elo.run_elo(_match("A", "B", 1, 0, bucket="world_cup"))[1]
    fr = elo.run_elo(_match("A", "B", 1, 0, bucket="friendly"))[1]
    # World Cup K=60 moves rating more than friendly K=20
    assert (wc["A"] - C.ELO_START) > (fr["A"] - C.ELO_START)


def test_draw_no_change_when_balanced():
    final = elo.run_elo(_match("A", "B", 1, 1))[1]
    assert final["A"] == pytest.approx(C.ELO_START)
    assert final["B"] == pytest.approx(C.ELO_START)


def test_convergence_on_real_history():
    import ingest
    final = elo.run_elo(ingest.build_history())[1]
    top = sorted(final.items(), key=lambda kv: -kv[1])[:8]
    top_names = {t for t, _ in top}
    # Perennial powers should sit near the top by mid-2026.
    assert len({"Brazil", "Argentina", "France", "Spain"} & top_names) >= 3
    assert all(e > 1900 for _, e in top)
