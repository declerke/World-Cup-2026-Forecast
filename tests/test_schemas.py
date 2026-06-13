"""Validate published JSON files against their contracts (requires a pipeline run)."""
import json

import pytest
from jsonschema import validate

import config as C
import schemas

pytestmark = pytest.mark.skipif(
    not (C.WEB_DATA / "meta.json").exists(),
    reason="pipeline outputs not generated yet",
)


@pytest.mark.parametrize("name", ["meta.json", "champion_odds.json", "groups.json",
                                  "matches.json", "bracket.json", "accuracy.json"])
def test_published_file_matches_schema(name):
    payload = json.loads((C.WEB_DATA / name).read_text(encoding="utf-8"))
    validate(instance=payload, schema=schemas.BY_FILE[name])


def test_champion_odds_sum_reasonable():
    payload = json.loads((C.WEB_DATA / "champion_odds.json").read_text(encoding="utf-8"))
    total = sum(t["p_champion"] for t in payload["teams"])
    assert total == pytest.approx(1.0, abs=1e-3)
    assert len(payload["teams"]) == 48


def test_match_details_present_and_valid():
    detail_dir = C.WEB_DATA / "match_detail"
    files = list(detail_dir.glob("*.json"))
    assert files, "expected at least one match_detail file"
    for f in files[:5]:
        d = json.loads(f.read_text(encoding="utf-8"))
        validate(instance=d, schema=schemas.MATCH_DETAIL)
        s = d["probs"]
        assert s["p_home"] + s["p_draw"] + s["p_away"] == pytest.approx(1.0, abs=1e-2)
