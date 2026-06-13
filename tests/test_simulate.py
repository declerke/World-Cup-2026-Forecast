"""Simulation invariants on a small seeded run (requires trained models)."""
import pytest

import config as C

pytestmark = pytest.mark.skipif(
    not (C.MODELS / "wdl_model.joblib").exists(),
    reason="models not trained yet",
)


@pytest.fixture(scope="module")
def sim_out():
    import artifacts
    import ingest
    import simulate
    art = artifacts.load()
    fixtures = ingest.load_fixtures(refresh=False)
    sim = simulate.Simulator(art, fixtures, n_sims=300, seed=C.SEED)
    return sim.run()


def test_champion_probs_sum_to_one(sim_out):
    # Probabilities are rounded to 4dp for display, so allow rounding drift.
    total = sum(d["p_champion"] for d in sim_out["teams"].values())
    assert total == pytest.approx(1.0, abs=0.01)


def test_all_48_teams_present(sim_out):
    assert len(sim_out["teams"]) == 48


def test_monotonic_advancement(sim_out):
    # P(reach R32) >= P(R16) >= P(QF) >= P(SF) >= P(final) >= P(champion)
    for d in sim_out["teams"].values():
        assert d["p_r32"] >= d["p_r16"] - 1e-9
        assert d["p_r16"] >= d["p_qf"] - 1e-9
        assert d["p_qf"] >= d["p_sf"] - 1e-9
        assert d["p_sf"] >= d["p_final"] - 1e-9
        assert d["p_final"] >= d["p_champion"] - 1e-9


def test_group_position_probs_sum_to_one(sim_out):
    for d in sim_out["teams"].values():
        s = d["p_win_group"] + d["p_runner_up"] + d["p_third"] + d["p_fourth"]
        assert s == pytest.approx(1.0, abs=0.001)


def test_twelve_group_winners_expected(sim_out):
    # Sum of P(win group) across all teams == 12 (one winner per group per sim)
    total = sum(d["p_win_group"] for d in sim_out["teams"].values())
    assert total == pytest.approx(12.0, abs=0.02)


def test_favorites_have_higher_champion_odds(sim_out):
    # A top-Elo side should out-rank a minnow in champion probability.
    teams = sim_out["teams"]
    strong = max(teams.values(), key=lambda d: d["elo"])
    weak = min(teams.values(), key=lambda d: d["elo"])
    assert strong["p_champion"] >= weak["p_champion"]
