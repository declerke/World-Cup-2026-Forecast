"""Ingest layer: pull and parse the three verified data sources.

Sources (all free, verified 2026-06-13):
  - martj42 international_results (GitHub raw CSV): full match history.
  - football-data.org v4: official WC 2026 fixtures, results, standings.

Raw payloads are cached under data/raw/. Re-download is controlled by the
`refresh` flag so tests and repeated runs don't hammer the network.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

import config as C
import team_names as tn


# ----------------------------------------------------------------------------
# Low-level fetch with retry
# ----------------------------------------------------------------------------
def _get(url: str, headers: dict | None = None, retries: int = 3) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # pragma: no cover - network
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


# ----------------------------------------------------------------------------
# martj42 history
# ----------------------------------------------------------------------------
def refresh_martj42() -> None:
    for url, name in (
        (C.MARTJ42_RESULTS_URL, "results.csv"),
        (C.MARTJ42_SHOOTOUTS_URL, "shootouts.csv"),
    ):
        r = _get(url)
        (C.RAW / name).write_bytes(r.content)


def load_results(refresh: bool = False) -> pd.DataFrame:
    """Full international match history with canonical names and parsed dates.

    Columns: date, home_team, away_team, home_score, away_score, tournament,
             city, country, neutral (bool), bucket.
    """
    path = C.RAW / "results.csv"
    if refresh or not path.exists():
        refresh_martj42()
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["home_team"] = df["home_team"].map(lambda s: tn.to_canonical(s, source="martj42"))
    df["away_team"] = df["away_team"].map(lambda s: tn.to_canonical(s, source="martj42"))
    df["bucket"] = df["tournament"].map(tn.tournament_bucket)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_shootouts(refresh: bool = False) -> pd.DataFrame:
    path = C.RAW / "shootouts.csv"
    if refresh or not path.exists():
        refresh_martj42()
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    for col in ("home_team", "away_team", "winner"):
        df[col] = df[col].map(lambda s: tn.to_canonical(s, source="martj42"))
    return df


# ----------------------------------------------------------------------------
# football-data.org WC 2026
# ----------------------------------------------------------------------------
def _fd_get(endpoint: str) -> dict:
    headers = {"X-Auth-Token": C.football_data_token()}
    url = f"{C.FOOTBALL_DATA_BASE}/{endpoint}"
    r = _get(url, headers=headers)
    return r.json()


def refresh_football_data() -> None:
    matches = _fd_get(f"competitions/{C.WC_COMPETITION_CODE}/matches?season={C.WC_SEASON}")
    (C.RAW / "fd_matches.json").write_text(
        json.dumps(matches, ensure_ascii=False), encoding="utf-8"
    )
    standings = _fd_get(f"competitions/{C.WC_COMPETITION_CODE}/standings?season={C.WC_SEASON}")
    (C.RAW / "fd_standings.json").write_text(
        json.dumps(standings, ensure_ascii=False), encoding="utf-8"
    )


# football-data.org stage -> our internal stage label + match-number ranges.
_STAGE_LABEL = {
    "GROUP_STAGE": "group",
    "LAST_32": "R32",
    "LAST_16": "R16",
    "QUARTER_FINALS": "QF",
    "SEMI_FINALS": "SF",
    "THIRD_PLACE": "3RD",
    "FINAL": "FINAL",
}


def load_fixtures(refresh: bool = False) -> pd.DataFrame:
    """Official WC 2026 fixtures with canonical names.

    Columns: match_id, utc_date, status, matchday, stage, group (letter or None),
             home_team, away_team, home_score, away_score, played (bool).
    Knockout fixtures may have null teams until the bracket resolves; those are
    placeholders we fill via simulation, so they are kept with None teams.
    """
    path = C.RAW / "fd_matches.json"
    if refresh or not path.exists():
        refresh_football_data()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for m in data["matches"]:
        ft = m.get("score", {}).get("fullTime", {})
        group = m.get("group")
        group_letter = group.replace("GROUP_", "") if group else None
        ht = m["homeTeam"].get("name")
        at = m["awayTeam"].get("name")
        rows.append({
            "match_id": m["id"],
            "utc_date": m["utcDate"],
            "status": m["status"],
            "matchday": m.get("matchday"),
            "stage": _STAGE_LABEL.get(m["stage"], m["stage"]),
            "group": group_letter,
            "home_team": tn.to_canonical(ht, source="fd") if ht else None,
            "away_team": tn.to_canonical(at, source="fd") if at else None,
            "home_score": ft.get("home"),
            "away_score": ft.get("away"),
            "played": m["status"] in ("FINISHED", "AWARDED"),
        })
    df = pd.DataFrame(rows).sort_values(["utc_date", "match_id"]).reset_index(drop=True)
    return df


def played_results(refresh: bool = False) -> pd.DataFrame:
    """Completed WC 2026 group matches, shaped like the martj42 history so they
    can be appended for Elo / form continuity even before martj42 catches up."""
    fx = load_fixtures(refresh=refresh)
    done = fx[(fx["played"]) & (fx["home_team"].notna())].copy()
    done = done[done["home_score"].notna()]
    out = pd.DataFrame({
        "date": pd.to_datetime(done["utc_date"]).dt.tz_localize(None).dt.normalize(),
        "home_team": done["home_team"],
        "away_team": done["away_team"],
        "home_score": done["home_score"].astype(int),
        "away_score": done["away_score"].astype(int),
        "tournament": "FIFA World Cup",
        "city": None,
        "country": None,
        "neutral": [t in ("United States", "Mexico", "Canada") for t in done["home_team"]],
        "bucket": "world_cup",
    })
    return out.reset_index(drop=True)


def build_history(refresh: bool = False) -> pd.DataFrame:
    """martj42 history + any WC 2026 results martj42 hasn't merged yet (deduped)."""
    hist = load_results(refresh=refresh)
    wc = played_results(refresh=refresh)
    combined = pd.concat([hist, wc], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score"],
        keep="first",
    )
    return combined.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    h = build_history()
    fx = load_fixtures()
    print(f"history rows: {len(h):,}  (latest {h['date'].max().date()})")
    print(f"fixtures: {len(fx)}  played: {int(fx['played'].sum())}")
