"""Feature engineering: leakage-safety and structural guarantees."""
import numpy as np
import pandas as pd
import pytest

import features as F
import config as C


def _toy_history():
    rows = [
        ("1995-01-01", "A", "B", 2, 0),
        ("1995-02-01", "B", "C", 1, 1),
        ("1995-03-01", "A", "C", 3, 1),
        ("1995-04-01", "C", "A", 0, 0),
        ("1995-05-01", "B", "A", 2, 1),
        ("1995-06-01", "A", "B", 1, 1),
    ]
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = True
    df["bucket"] = "friendly"
    df["tournament"] = "Friendly"
    df["city"] = None
    df["country"] = None
    return df


def test_feature_columns_count_and_order():
    assert len(F.FEATURE_COLUMNS) == 30
    assert F.FEATURE_COLUMNS[0] == "elo_home"
    assert "elo_diff" in F.FEATURE_COLUMNS


def test_first_match_has_default_form():
    train, _, _ = F.build(_toy_history().assign(date=lambda d: d["date"]))
    # all toy matches are >= TRAIN_FROM (1995 > 1990), so first row present
    first = train.iloc[0]
    # No prior history -> default form (0.5 win rate)
    assert first["form10_win_h"] == 0.5
    assert first["form10_win_a"] == 0.5
    assert first["elo_home"] == C.ELO_START
    assert first["elo_away"] == C.ELO_START


def test_no_future_leakage():
    """A feature row must be identical whether or not later matches exist.

    Build features on the full history, then on a truncated history ending at
    match k; the feature row for match k must be byte-identical (it may only use
    the past).
    """
    hist = _toy_history()
    full, _, _ = F.build(hist)
    for k in range(1, len(hist)):
        truncated, _, _ = F.build(hist.iloc[: k + 1].reset_index(drop=True))
        row_full = full.iloc[k][F.FEATURE_COLUMNS].astype(float).to_numpy()
        row_trunc = truncated.iloc[k][F.FEATURE_COLUMNS].astype(float).to_numpy()
        np.testing.assert_allclose(row_full, row_trunc, rtol=1e-9,
                                   err_msg=f"leakage at match {k}")


def test_outcome_labels_match_scores():
    train, _, _ = F.build(_toy_history())
    # match 0: A 2-0 B -> home_win
    assert train.iloc[0]["outcome"] == "home_win"
    # match 1: B 1-1 C -> draw
    assert train.iloc[1]["outcome"] == "draw"


def test_form_reflects_prior_result():
    train, _, _ = F.build(_toy_history())
    # By match 2 (A vs C), A has played one match (won) -> win rate 1.0
    assert train.iloc[2]["form10_win_h"] == 1.0


def test_features_for_pair_uses_final_state():
    _, states, h2h = F.build(_toy_history())
    feat = F.features_for_pair("A", "B", neutral=True, states=states, h2h_log=h2h)
    assert set(F.FEATURE_COLUMNS).issubset(feat.keys())
    assert feat["neutral"] == 1
