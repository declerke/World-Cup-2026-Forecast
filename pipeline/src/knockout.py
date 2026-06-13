"""Knockout-tie resolution.

No draws in knockout rounds. We sample the 90-minute outcome from the classifier;
a draw goes to extra-time/penalties, won by the (neutral-venue) Elo favourite with
probability We = 1/(1+10**(-elo_diff/400)). Simple, defensible, documented in the
model card.
"""
from __future__ import annotations

import numpy as np


def shootout_prob(elo_a: float, elo_b: float) -> float:
    """P(a wins a shootout/extra-time) from neutral-venue Elo difference."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))


def advance_prob(p_a_win: float, p_draw: float, elo_a: float, elo_b: float) -> float:
    """Analytic P(a advances) = P(a win) + P(draw) * shootout edge."""
    return p_a_win + p_draw * shootout_prob(elo_a, elo_b)


def sample_winner(rng: np.random.Generator, p_a_win, p_draw, p_b_win,
                  elo_a, elo_b) -> int:
    """Return 0 if a advances, 1 if b advances (single sample)."""
    r = rng.random()
    if r < p_a_win:
        return 0
    if r < p_a_win + p_b_win:
        return 1
    # draw -> shootout
    return 0 if rng.random() < shootout_prob(elo_a, elo_b) else 1
