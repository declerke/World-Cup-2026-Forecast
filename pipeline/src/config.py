"""Central configuration: paths, constants, RNG policy, tournament metadata.

All file IO in this project goes through UTF-8 (Windows defaults to cp1252,
which corrupts team names like Curaçao / Côte d'Ivoire).
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipeline"
DATA = PIPELINE / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FROZEN = DATA / "frozen"          # committed to git
MODELS = PIPELINE / "models"
SCHEMAS = PIPELINE / "schemas"
WEB_DATA = ROOT / "web" / "public" / "data"   # committed; consumed by frontend

for _d in (RAW, PROCESSED, FROZEN, MODELS, WEB_DATA):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data sources (all verified free, 2026-06-13)
# ----------------------------------------------------------------------------
MARTJ42_RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
MARTJ42_SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
WC_COMPETITION_CODE = "WC"          # id 2000
WC_SEASON = 2026

def football_data_token() -> str:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    tok = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
    if not tok:
        raise RuntimeError(
            "FOOTBALL_DATA_TOKEN missing. Set it in .env (local) or repo secret (CI)."
        )
    return tok

# ----------------------------------------------------------------------------
# Tournament metadata
# ----------------------------------------------------------------------------
TOURNAMENT_START = date(2026, 6, 11)
TOURNAMENT_END = date(2026, 7, 19)

# ----------------------------------------------------------------------------
# Modeling constants
# ----------------------------------------------------------------------------
SEED = 42
N_SIMULATIONS = 10_000
TRAIN_FROM = "1990-01-01"          # Elo warm-started on full 1872+ history
TEST_FROM = "2024-01-01"           # held-out window, never tuned on
TEST_TO = "2026-06-10"
TIME_DECAY_HALF_LIFE_YEARS = 4.0
FORM_WINDOWS = (5, 10)
MAX_REST_DAYS = 60

# Elo engine
ELO_START = 1500.0
ELO_HOME_ADVANTAGE = 100.0

# K-factor by tournament importance bucket (see team_names.tournament_bucket)
ELO_K = {
    "world_cup": 60,
    "continental_final": 50,
    "qualifier": 40,
    "other_tournament": 30,
    "friendly": 20,
}

# Poisson goal lambda clamp
LAMBDA_MIN = 0.2
LAMBDA_MAX = 4.5
MAX_GOALS = 10                     # scoreline matrix is (MAX_GOALS+1) square


def run_seed(run_date: date | None = None) -> int:
    """Deterministic per-day seed so a given day's forecast is reproducible."""
    run_date = run_date or date.today()
    return int(run_date.strftime("%Y%m%d"))
