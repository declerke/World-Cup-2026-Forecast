"""Emit the seven JSON data contracts the frontend consumes.

Each file is validated against its schema before being written to
web/public/data/. Upcoming fixtures with known teams also get a match_detail
file (probabilities, top scorelines, SHAP explanation, form, h2h, Elo history).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from jsonschema import validate

import config as C
import predict
import freeze
import team_names as tn
import bracket as B
import venues
import sys
sys.path.insert(0, str(C.SCHEMAS))
import schemas  # noqa: E402

FEATURE_LABELS = {
    "elo_diff": "Elo rating gap", "elo_home": "home Elo", "elo_away": "away Elo",
    "neutral": "neutral venue", "home_is_host": "host nation at home",
    "form10_win_h": "home recent win rate", "form10_win_a": "away recent win rate",
    "form5_win_h": "home last-5 form", "form5_win_a": "away last-5 form",
    "form10_gf_h": "home scoring form", "form10_gf_a": "away scoring form",
    "form10_ga_h": "home defensive form", "form10_ga_a": "away defensive form",
    "form10_oppelo_h": "home strength of schedule", "form10_oppelo_a": "away strength of schedule",
    "h2h_home_winrate": "head-to-head record", "h2h_mean_gd": "head-to-head goal margin",
    "elo_trend_h": "home momentum (1yr Elo)", "elo_trend_a": "away momentum (1yr Elo)",
    "rest_h": "home rest days", "rest_a": "away rest days", "importance": "match importance",
    "alt_gap_home": "altitude vs home's norm", "alt_gap_away": "altitude vs away's norm",
    "form5_draw_h": "home draw tendency", "form5_draw_a": "away draw tendency",
    "form10_draw_h": "home draw tendency (10)", "form10_draw_a": "away draw tendency (10)",
    "form5_gf_h": "home last-5 scoring", "form5_gf_a": "away last-5 scoring",
    "form5_ga_h": "home last-5 defense", "form5_ga_a": "away last-5 defense",
}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=C.ROOT, text=True).strip()
    except Exception:
        return "uncommitted"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_safe(obj):
    """Recursively convert NaN/inf and numpy scalars to JSON-valid values.

    Browsers reject the `NaN` token Python's json emits by default, so every
    payload is sanitised (NaN/inf -> null) before validation and writing.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _write(name: str, payload: dict):
    payload = _json_safe(payload)
    schema = schemas.BY_FILE.get(name)
    if schema:
        validate(instance=payload, schema=schema)
    (C.WEB_DATA / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")


class Publisher:
    def __init__(self, art, fixtures, sim_out, metrics, model_version):
        self.art = art
        self.states = art["states"]
        self.h2h = art["h2h"]
        self.timeline = art["elo_timeline"]
        self.baselines = art.get("altitude_baselines", {})
        self.fixtures = fixtures
        self.sim = sim_out
        self.metrics = metrics
        self.version = model_version
        self._explainer = None
        self.predictions = {}   # match_id -> wdl + score, for freeze

    # -- per-fixture prediction ---------------------------------------------
    def _fixture_prediction(self, m):
        home, away = m["home_team"], m["away_team"]
        neutral = predict.is_neutral(home)
        v_alt = venues.wc2026_altitude(home, away)
        wdl = predict.wdl(home, away, neutral=neutral, states=self.states,
                          h2h=self.h2h, importance=4,
                          baselines=self.baselines, venue_altitude=v_alt)
        mat = predict.scoreline_matrix(home, away, neutral=neutral, states=self.states,
                                       h2h=self.h2h, importance=4, wdl_probs=wdl,
                                       baselines=self.baselines, venue_altitude=v_alt)
        tops = predict.top_scorelines(mat, 5)
        return wdl, mat, tops

    def _venue_context(self, home, away):
        """Venue city + altitude note for the match page (None if sea-level)."""
        city = venues.wc2026_venue_city(home, away)
        if city is None:
            return None
        alt = venues.wc2026_altitude(home, away)
        gap_home = max(0.0, alt - self.baselines.get(home, 0.0))
        gap_away = max(0.0, alt - self.baselines.get(away, 0.0))
        # The side with the smaller ascent is the more acclimatised.
        if abs(gap_home - gap_away) < 50:
            favours = None
        else:
            favours = home if gap_home < gap_away else away
        return {"city": city, "altitude_m": int(alt),
                "ascent_home_m": int(gap_home), "ascent_away_m": int(gap_away),
                "favours": favours}

    # -- SHAP ----------------------------------------------------------------
    def _shap_top(self, m, favorite_class: int):
        import features as F
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(predict.load_models()["wdl"])
        neutral = predict.is_neutral(m["home_team"])
        v_alt = venues.wc2026_altitude(m["home_team"], m["away_team"])
        feat = F.features_for_pair(m["home_team"], m["away_team"], neutral=neutral,
                                   states=self.states, h2h_log=self.h2h, importance=4,
                                   baselines=self.baselines, venue_altitude=v_alt)
        x = np.array([[feat[c] for c in F.FEATURE_COLUMNS]], dtype=float)
        sv = self._explainer.shap_values(x)
        if isinstance(sv, list):              # list per class
            arr = sv[favorite_class][0]
        else:                                 # (1, n_feat, n_class)
            arr = sv[0, :, favorite_class]
        order = np.argsort(np.abs(arr))[::-1][:6]
        out = []
        for i in order:
            col = F.FEATURE_COLUMNS[i]
            out.append({"feature": FEATURE_LABELS.get(col, col),
                        "impact": round(float(arr[i]), 4),
                        "direction": "favors" if arr[i] > 0 else "against"})
        return out

    def _form_strip(self, team, n=10):
        st = self.states.get(team)
        if not st:
            return []
        letters = {1.0: "W", 0.5: "D", 0.0: "L"}
        return [letters[r] for r in list(st.results)[-n:]]

    def _elo_history(self, team, months=24):
        tl = self.timeline[self.timeline["team"] == team].copy()
        if tl.empty:
            return []
        tl = tl.set_index("date").sort_index()
        monthly = tl["elo"].resample("MS").last().dropna().tail(months)
        return [{"date": d.strftime("%Y-%m"), "elo": round(float(v), 1)}
                for d, v in monthly.items()]

    # -- emitters ------------------------------------------------------------
    def emit_meta(self):
        _write("meta.json", {
            "generated_at_utc": _now(), "model_version": self.version,
            "n_sims": self.sim["n_sims"],
            "training_rows": int(len(self.art["train"])),
            "metrics": {k: v for k, v in self.metrics.items()
                        if isinstance(v, (int, float, bool))},
        })

    def emit_champion_odds(self):
        prev = {}
        path = C.WEB_DATA / "champion_odds.json"
        if path.exists():
            for t in json.loads(path.read_text(encoding="utf-8")).get("teams", []):
                prev[t["team"]] = t["p_champion"]
        teams = sorted(self.sim["teams"].values(), key=lambda d: -d["p_champion"])
        out = []
        for d in teams:
            delta = (round(d["p_champion"] - prev[d["team"]], 4)
                     if d["team"] in prev else None)
            out.append({"team": d["team"], "group": d["group"],
                        "p_champion": d["p_champion"], "p_final": d["p_final"],
                        "p_sf": d["p_sf"], "p_qf": d["p_qf"], "p_r16": d["p_r16"],
                        "p_r32": d["p_r32"], "elo": d["elo"],
                        "delta_vs_yesterday": delta})
        _write("champion_odds.json", {"generated_at_utc": _now(),
                                      "model_version": self.version, "teams": out})

    def emit_groups(self):
        standings = self._current_standings()
        groups = []
        for g, team_list in tn.GROUPS.items():
            rows = []
            for t in team_list:
                s = self.sim["teams"][t]
                st = standings.get(t, {})
                rows.append({
                    "team": t, "elo": s["elo"],
                    "played": st.get("played", 0), "points": st.get("points", 0),
                    "gd": st.get("gd", 0), "gf": st.get("gf", 0),
                    "p_win_group": s["p_win_group"], "p_runner_up": s["p_runner_up"],
                    "p_third": s["p_third"],
                    "p_advance": round(min(1.0, s["p_r32"]), 4),
                })
            rows.sort(key=lambda r: (-r["points"], -r["gd"], -r["gf"], -r["p_advance"]))
            groups.append({"group": g, "teams": rows})
        _write("groups.json", {"generated_at_utc": _now(),
                               "model_version": self.version, "groups": groups})

    def _current_standings(self):
        played = self.fixtures[(self.fixtures["stage"] == "group") &
                               (self.fixtures["played"]) &
                               (self.fixtures["home_score"].notna())]
        tbl = {}
        for _, m in played.iterrows():
            h, a = m["home_team"], m["away_team"]
            hg, ag = int(m["home_score"]), int(m["away_score"])
            for t in (h, a):
                tbl.setdefault(t, {"played": 0, "points": 0, "gd": 0, "gf": 0})
            tbl[h]["played"] += 1; tbl[a]["played"] += 1
            tbl[h]["gf"] += hg; tbl[a]["gf"] += ag
            tbl[h]["gd"] += hg - ag; tbl[a]["gd"] += ag - hg
            if hg > ag:
                tbl[h]["points"] += 3
            elif hg < ag:
                tbl[a]["points"] += 3
            else:
                tbl[h]["points"] += 1; tbl[a]["points"] += 1
        return tbl

    def emit_matches(self):
        out = []
        for _, m in self.fixtures.iterrows():
            row = {"match_id": int(m["match_id"]), "stage": m["stage"],
                   "status": m["status"], "group": m["group"],
                   "utc_date": m["utc_date"], "home_team": m["home_team"],
                   "away_team": m["away_team"], "matchday": m["matchday"]}
            if m["played"] and m["home_score"] is not None:
                row["score"] = {"home": int(m["home_score"]), "away": int(m["away_score"])}
            elif m["home_team"] and m["away_team"]:
                wdl, _mat, tops = self._fixture_prediction(m)
                row["probs"] = {"p_home": round(wdl[0], 4), "p_draw": round(wdl[1], 4),
                                "p_away": round(wdl[2], 4)}
                row["most_likely_score"] = {"home": tops[0]["home"], "away": tops[0]["away"]}
                vc = self._venue_context(m["home_team"], m["away_team"])
                if vc:
                    row["venue"] = vc
            out.append(row)
        _write("matches.json", {"generated_at_utc": _now(),
                                "model_version": self.version, "matches": out})

    def emit_match_details(self):
        detail_dir = C.WEB_DATA / "match_detail"
        detail_dir.mkdir(exist_ok=True)
        upcoming = self.fixtures[(~self.fixtures["played"]) &
                                 (self.fixtures["home_team"].notna()) &
                                 (self.fixtures["away_team"].notna())]
        for _, m in upcoming.iterrows():
            wdl, mat, tops = self._fixture_prediction(m)
            fav = int(np.argmax(wdl))
            self.predictions[int(m["match_id"])] = {
                "p_home": wdl[0], "p_draw": wdl[1], "p_away": wdl[2],
                "score_home": tops[0]["home"], "score_away": tops[0]["away"]}
            detail = {
                "match_id": int(m["match_id"]),
                "home_team": m["home_team"], "away_team": m["away_team"],
                "stage": m["stage"], "group": m["group"], "utc_date": m["utc_date"],
                "probs": {"p_home": round(wdl[0], 4), "p_draw": round(wdl[1], 4),
                          "p_away": round(wdl[2], 4)},
                "top_scorelines": tops,
                "shap": self._shap_top(m, fav),
                "form": {"home": self._form_strip(m["home_team"]),
                         "away": self._form_strip(m["away_team"])},
                "h2h": self._h2h_summary(m["home_team"], m["away_team"]),
                "elo_history": {"home": self._elo_history(m["home_team"]),
                                "away": self._elo_history(m["away_team"])},
                "venue": self._venue_context(m["home_team"], m["away_team"]),
            }
            detail = _json_safe(detail)
            validate(instance=detail, schema=schemas.MATCH_DETAIL)
            (detail_dir / f"{int(m['match_id'])}.json").write_text(
                json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8")

    def _h2h_summary(self, home, away):
        key = tuple(sorted((home, away)))
        log = self.h2h.get(key, [])[-10:]
        hw = aw = d = 0
        for ht, hg, ag in log:
            if hg == ag:
                d += 1
            elif (ht == home) == (hg > ag):
                hw += 1
            else:
                aw += 1
        return {"meetings": len(log), "home_wins": hw, "away_wins": aw, "draws": d}

    def emit_bracket(self):
        slots = {str(k): v for k, v in self.sim["slots"].items()}
        _write("bracket.json", {"generated_at_utc": _now(),
                                "model_version": self.version,
                                "structure": {str(k): list(v) for k, v in B.KNOCKOUT.items()},
                                "slots": slots})

    def emit_accuracy(self):
        scores = freeze._read_csv(freeze.SCORES_PATH, freeze.SCORE_FIELDS)
        log = freeze._read_csv(freeze.LOG_PATH, freeze.LOG_FIELDS)
        summary, history, calibration, receipts = self._accuracy_tables(scores, log)
        _write("accuracy.json", {"generated_at_utc": _now(),
                                 "model_version": self.version, "summary": summary,
                                 "history": history, "calibration": calibration,
                                 "receipts": receipts})

    def _accuracy_tables(self, scores, log):
        if not scores:
            return ({"n_scored": 0, "mean_brier": None, "mean_logloss": None,
                     "favorite_accuracy": None, "exact_accuracy": None}, [], [], [])
        df = pd.DataFrame(scores)
        for c in ("brier", "logloss", "favorite_correct", "exact_correct"):
            df[c] = pd.to_numeric(df[c])
        summary = {
            "n_scored": int(len(df)),
            "mean_brier": round(float(df["brier"].mean()), 4),
            "mean_logloss": round(float(df["logloss"].mean()), 4),
            "favorite_accuracy": round(float(df["favorite_correct"].mean()), 4),
            "exact_accuracy": round(float(df["exact_correct"].mean()), 4),
        }
        df = df.sort_values("kickoff_utc")
        df["cum_brier"] = df["brier"].expanding().mean()
        df["cum_fav"] = df["favorite_correct"].expanding().mean()
        history = [{"match": i + 1, "kickoff": r["kickoff_utc"],
                    "cum_brier": round(r["cum_brier"], 4),
                    "cum_favorite_accuracy": round(r["cum_fav"], 4)}
                   for i, (_, r) in enumerate(df.iterrows())]
        # calibration bins on favourite probability
        df["fav_prob"] = df[["p_home", "p_draw", "p_away"]].astype(float).max(axis=1)
        bins = np.linspace(0.33, 1.0, 8)
        calibration = []
        for i in range(len(bins) - 1):
            m = (df["fav_prob"] >= bins[i]) & (df["fav_prob"] < bins[i + 1])
            if m.sum() > 0:
                calibration.append({"bin": round((bins[i] + bins[i + 1]) / 2, 3),
                                    "predicted": round(float(df.loc[m, "fav_prob"].mean()), 4),
                                    "actual": round(float(df.loc[m, "favorite_correct"].mean()), 4),
                                    "count": int(m.sum())})
        receipts = df[["match_id", "home", "away", "p_home", "p_draw", "p_away",
                       "pred_score_home", "pred_score_away", "actual_home",
                       "actual_away", "outcome", "favorite_correct"]].tail(50).to_dict("records")
        return summary, history, calibration, receipts

    def run(self):
        # match_details first (populates self.predictions used by freeze)
        self.emit_match_details()
        self.emit_meta()
        self.emit_champion_odds()
        self.emit_groups()
        self.emit_matches()
        self.emit_bracket()
        self.emit_accuracy()
        return self.predictions
