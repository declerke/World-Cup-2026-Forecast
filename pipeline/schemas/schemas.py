"""JSON Schemas (draft-2020-12) for the seven published data contracts.

The publisher validates every emitted file against these before writing, so the
frontend can rely on the shapes. Kept intentionally structural (key presence +
types) rather than exhaustively constraining every numeric range.
"""

_envelope = {"generated_at_utc": {"type": "string"}, "model_version": {"type": "string"}}

META = {
    "type": "object",
    "required": ["generated_at_utc", "model_version", "n_sims", "metrics"],
    "properties": {
        **_envelope,
        "n_sims": {"type": "integer"},
        "training_rows": {"type": "integer"},
        "metrics": {"type": "object"},
    },
}

CHAMPION_ODDS = {
    "type": "object",
    "required": ["generated_at_utc", "teams"],
    "properties": {
        **_envelope,
        "teams": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["team", "group", "p_champion", "p_final", "p_sf",
                             "p_qf", "p_r16", "p_r32"],
                "properties": {
                    "team": {"type": "string"}, "group": {"type": "string"},
                    "p_champion": {"type": "number"}, "p_final": {"type": "number"},
                    "p_sf": {"type": "number"}, "p_qf": {"type": "number"},
                    "p_r16": {"type": "number"}, "p_r32": {"type": "number"},
                    "delta_vs_yesterday": {"type": ["number", "null"]},
                },
            },
        },
    },
}

GROUPS = {
    "type": "object",
    "required": ["generated_at_utc", "groups"],
    "properties": {
        **_envelope,
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["group", "teams"],
                "properties": {
                    "group": {"type": "string"},
                    "teams": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
    },
}

MATCHES = {
    "type": "object",
    "required": ["generated_at_utc", "matches"],
    "properties": {
        **_envelope,
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["match_id", "stage", "status", "utc_date"],
                "properties": {
                    "match_id": {"type": "integer"},
                    "stage": {"type": "string"}, "status": {"type": "string"},
                    "group": {"type": ["string", "null"]},
                    "utc_date": {"type": "string"},
                    "home_team": {"type": ["string", "null"]},
                    "away_team": {"type": ["string", "null"]},
                },
            },
        },
    },
}

MATCH_DETAIL = {
    "type": "object",
    "required": ["match_id", "home_team", "away_team", "probs"],
    "properties": {
        "match_id": {"type": "integer"},
        "home_team": {"type": "string"}, "away_team": {"type": "string"},
        "probs": {
            "type": "object",
            "required": ["p_home", "p_draw", "p_away"],
            "properties": {"p_home": {"type": "number"}, "p_draw": {"type": "number"},
                           "p_away": {"type": "number"}},
        },
        "top_scorelines": {"type": "array"},
        "shap": {"type": "array"},
    },
}

BRACKET = {
    "type": "object",
    "required": ["generated_at_utc", "slots"],
    "properties": {**_envelope, "slots": {"type": "object"}},
}

ACCURACY = {
    "type": "object",
    "required": ["generated_at_utc", "summary", "history"],
    "properties": {
        **_envelope,
        "summary": {"type": "object"},
        "history": {"type": "array"},
        "calibration": {"type": "array"},
        "receipts": {"type": "array"},
    },
}

BY_FILE = {
    "meta.json": META,
    "champion_odds.json": CHAMPION_ODDS,
    "groups.json": GROUPS,
    "matches.json": MATCHES,
    "bracket.json": BRACKET,
    "accuracy.json": ACCURACY,
}
