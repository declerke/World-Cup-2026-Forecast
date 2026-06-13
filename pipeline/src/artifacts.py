"""Build-once artifacts shared across train / simulate / publish.

Caches the walk-forward training matrix, final team states, and h2h log so a
single pipeline run doesn't recompute the 49k-match pass multiple times.
"""
from __future__ import annotations

import joblib

import config as C
import ingest
import elo
import features

_CACHE = C.PROCESSED / "artifacts.joblib"


def build(refresh: bool = False) -> dict:
    history = ingest.build_history(refresh=refresh)
    matches_elo, final_elo, timeline = elo.run_elo(history)
    train, states, h2h, baselines = features.build(history)
    art = {
        "history": history,
        "final_elo": final_elo,
        "elo_timeline": timeline,
        "train": train,
        "states": states,
        "h2h": h2h,
        "altitude_baselines": baselines,
    }
    joblib.dump(art, _CACHE)
    return art


def load(refresh: bool = False) -> dict:
    if refresh or not _CACHE.exists():
        return build(refresh=refresh)
    return joblib.load(_CACHE)
