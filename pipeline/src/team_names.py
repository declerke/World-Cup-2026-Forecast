"""Canonical team-name registry and cross-source name resolution.

Canonical convention = martj42 (the training dataset's names). The
football-data.org fixtures use a few different spellings; we map them here.
Every mapping below was verified against the live martj42 results.csv and the
football-data.org WC 2026 fixtures on 2026-06-13.

A hard failure (KeyError) is raised on any unmapped name — we never silently
drop a team, because that would corrupt the simulation.
"""
from __future__ import annotations

import unicodedata

# football-data.org spelling  ->  canonical (martj42) spelling
# Only the genuine differences are listed; identical names pass through.
FD_TO_CANONICAL = {
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    # Curaçao / Ivory Coast match once unicode is normalised consistently.
}

# The 48 qualified teams in their 12 official groups, canonical names.
# Derived from football-data.org WC 2026 fixtures (group field), then mapped.
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Turkey", "Australia"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Reverse index: canonical team -> group letter
TEAM_GROUP: dict[str, str] = {
    team: g for g, teams in GROUPS.items() for team in teams
}

ALL_TEAMS: list[str] = [t for teams in GROUPS.values() for t in teams]


def _nfc(s: str) -> str:
    """Normalise unicode so 'Curaçao' compares equal regardless of source."""
    return unicodedata.normalize("NFC", s).strip()


def to_canonical(name: str, *, source: str = "fd") -> str:
    """Resolve a source name to the canonical (martj42) spelling.

    source='fd'      -> football-data.org names
    source='martj42' -> already canonical (only normalised)
    """
    name = _nfc(name)
    if source == "fd":
        return FD_TO_CANONICAL.get(name, name)
    return name


def tournament_bucket(tournament: str) -> str:
    """Map a martj42 `tournament` value to an Elo K-factor bucket.

    Buckets: world_cup, continental_final, qualifier, other_tournament, friendly.
    """
    t = tournament.lower()
    if t == "friendly":
        return "friendly"
    if "fifa world cup" in t and "qualification" not in t:
        return "world_cup"
    if "qualification" in t or "qualifier" in t:
        return "qualifier"
    # Major continental + intercontinental finals tournaments
    continental = (
        "uefa euro", "copa américa", "copa america", "african cup of nations",
        "afc asian cup", "gold cup", "confederations cup",
        "uefa nations league", "concacaf nations league finals",
    )
    if any(c in t for c in continental) and "qualification" not in t:
        return "continental_final"
    return "other_tournament"
