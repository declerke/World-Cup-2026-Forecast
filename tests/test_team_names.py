"""Team-name resolution: all 48 qualified teams resolve from every source."""
import json

import pandas as pd
import pytest

import config as C
import team_names as tn


def test_48_teams_in_12_groups():
    assert len(tn.GROUPS) == 12
    assert len(tn.ALL_TEAMS) == 48
    assert all(len(v) == 4 for v in tn.GROUPS.values())
    assert len(set(tn.ALL_TEAMS)) == 48   # no duplicates


def test_all_teams_resolve_from_martj42():
    df = pd.read_csv(C.RAW / "results.csv", encoding="utf-8")
    names = {tn._nfc(x) for x in set(df["home_team"]) | set(df["away_team"])}
    missing = [t for t in tn.ALL_TEAMS if tn._nfc(t) not in names]
    assert missing == [], f"unmapped in martj42: {missing}"


def test_all_fixtures_resolve_to_a_group():
    data = json.loads((C.RAW / "fd_matches.json").read_text(encoding="utf-8"))
    unresolved = []
    for m in data["matches"]:
        if m["stage"] != "GROUP_STAGE":
            continue
        for side in ("homeTeam", "awayTeam"):
            name = m[side].get("name")
            if name:
                canon = tn.to_canonical(name, source="fd")
                if canon not in tn.TEAM_GROUP:
                    unresolved.append((name, canon))
    assert unresolved == [], f"unresolved fd names: {unresolved}"


def test_tournament_buckets():
    assert tn.tournament_bucket("Friendly") == "friendly"
    assert tn.tournament_bucket("FIFA World Cup") == "world_cup"
    assert tn.tournament_bucket("FIFA World Cup qualification") == "qualifier"
    assert tn.tournament_bucket("UEFA Euro") == "continental_final"
    assert tn.tournament_bucket("UEFA Euro qualification") == "qualifier"


def test_explicit_mappings():
    assert tn.to_canonical("Czechia") == "Czech Republic"
    assert tn.to_canonical("Congo DR") == "DR Congo"
    assert tn.to_canonical("Cape Verde Islands") == "Cape Verde"
    assert tn.to_canonical("Brazil") == "Brazil"   # passthrough
