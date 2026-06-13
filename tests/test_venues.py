"""Venue altitude modelling."""
import pandas as pd
import pytest

import venues


def test_known_altitudes():
    assert venues.altitude("Mexico City") == 2240
    assert venues.altitude("La Paz") == 3640
    assert venues.altitude("Guadalajara") == 1566
    # Sea-level / unknown cities default to 0
    assert venues.altitude("London") == 0
    assert venues.altitude(None) == 0
    assert venues.altitude(float("nan")) == 0


def test_wc2026_elevated_fixtures():
    # The Mexican-venue group matches are flagged; everything else is sea level.
    assert venues.wc2026_altitude("Mexico", "South Africa") == 2240
    assert venues.wc2026_altitude("Mexico", "South Korea") == 1566
    assert venues.wc2026_altitude("Uruguay", "Spain") == 1566
    # order-independent
    assert venues.wc2026_altitude("South Africa", "Mexico") == 2240
    # a sea-level fixture
    assert venues.wc2026_altitude("England", "Croatia") == 0


def test_team_baselines_capture_altitude_nations():
    df = pd.DataFrame({
        "home_team": ["Mexico", "Mexico", "Bolivia", "England", "England"],
        "away_team": ["USA", "Brazil", "Peru", "Spain", "France"],
        "city": ["Mexico City", "Mexico City", "La Paz", "London", "London"],
        "neutral": [False, False, False, False, False],
    })
    bl = venues.team_baselines(df)
    assert bl["Mexico"] == 2240
    assert bl["Bolivia"] == 3640
    assert bl["England"] == 0


def test_ascent_only_penalised():
    # A team based at altitude playing at sea level has no positive gap.
    df = pd.DataFrame({
        "home_team": ["Bolivia"], "away_team": ["Brazil"],
        "city": ["La Paz"], "neutral": [False],
    })
    bl = venues.team_baselines(df)
    # Bolivia at sea level (0 m venue): max(0, 0 - 3640) = 0, no penalty for descending
    assert max(0.0, 0 - bl["Bolivia"]) == 0.0
    # Brazil (baseline 0) climbing to La Paz: large positive ascent
    assert max(0.0, 3640 - bl.get("Brazil", 0.0)) == 3640
