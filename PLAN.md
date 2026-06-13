# World Cup 2026 Forecast — Implementation Plan

**Status:** Approved by Ian 2026-06-12 (Approach A + Poisson scoreline layer).
**Planner:** Fable 5. **Executor:** Sonnet (this document is your spec — follow it exactly; where it says VERIFY, verify before coding).
**Project folder:** `C:/Users/Administrator/OneDrive/Luxdev/worldcup-2026-forecast/`
**GitHub repo:** `declerke/World-Cup-2026-Forecast` (public)
**UI brand name:** **CupCast 2026**
**Deploy target:** Vercel (static React app + JSON data files). Pipeline runs via GitHub Actions daily cron.

---

## 1. Goal & success criteria

A daily-updating, publicly deployed ML forecast of the 2026 FIFA World Cup (June 11 – July 19; 48 teams, 12 groups, 104 matches, new round of 32):

1. Per-match win/draw/loss probabilities + most-likely scoreline for every remaining fixture.
2. 10,000-run Monte Carlo tournament simulation → champion odds and per-team advancement probabilities (R32/R16/QF/SF/Final/Champion).
3. **Self-grading accuracy tracker**: every prediction is frozen pre-kickoff and committed to the repo; the site publicly scores itself (Brier score, log loss, calibration) as results come in. This is the headline showcase feature — it only works if predictions are demonstrably made *before* matches.
4. Zero running cost. No paid APIs, no backend servers, real data only.

Honest framing everywhere: probabilities, not picks. State-of-the-art match prediction is ~55–60% accurate; the model card in the README must say so.

---

## 2. Architecture

```
GitHub Actions (daily cron 09:00 UTC)
 └─ run_pipeline.py
     1. ingest      → match history (martj42), Elo cross-check (eloratings.net),
                       official WC fixtures/results/standings (football-data.org)
     2. elo         → compute World Football Elo for all teams from full 1872+ history
     3. features    → leakage-safe feature matrix (walk-forward)
     4. train       → XGBoost W/D/L classifier (retrain daily on all data) + eval report
     5. poisson     → expected-goals model → scoreline matrices
     6. freeze      → log predictions for upcoming fixtures (append-only, committed)
     7. simulate    → 10,000 Monte Carlo tournaments (seeded)
     8. publish     → web/public/data/*.json (schema contracts in §10)
     9. score       → grade frozen predictions against new results
 └─ commit + push JSON/logs → Vercel auto-deploys
```

Two halves, one repo: `pipeline/` (Python) and `web/` (React). The JSON schema contracts in §10 are the only interface between them — neither side may deviate from them.

---

## 3. Data sources (all verified 2026-06-12, all free)

| # | Source | Use | Access | Fallback |
|---|--------|-----|--------|----------|
| 1 | martj42 international results | Training history: ~49,400 men's full internationals 1872–present, **updated daily, includes WC 2026 results as played** | Primary: raw CSV from GitHub `https://github.com/martj42/international_results` (`results.csv`, `shootouts.csv`). **VERIFY raw URLs at build time.** | Kaggle API (`martj42/international-football-results-from-1872-to-2017`) — needs KAGGLE creds, avoid if GitHub raw works |
| 2 | eloratings.net | **Cross-check only** for our self-computed Elo (spot-check ~10 major teams within ±25 pts) | Plain TSV over HTTPS, no auth. **VERIFY exact TSV URL by inspecting the site's network calls.** | Skip cross-check if unreachable (warn, don't fail) |
| 3 | football-data.org v4 | Official 2026 fixture list, results, group standings, team names/groups | Free tier, token in env `FOOTBALL_DATA_TOKEN`, 10 calls/min. Competition code `WC` (id 2000). Endpoints: `/v4/competitions/WC/matches`, `/v4/competitions/WC/standings`, `/v4/competitions/WC/teams`. **VERIFY season=2026 payload shape before coding the parser.** | API-Football free plan (`league=1&season=2026`, 100 req/day) — implement only if football-data.org proves unreliable |
| 4 | StatsBomb open data | **DEFERRED post-launch** (editorial xG enrichment). Not in v1. | — | — |

**Critical name-mapping task:** martj42, eloratings, and football-data.org use different team-name conventions ("USA" / "United States" / "USA"; "South Korea" / "Korea Republic"; "Iran" / "IR Iran"; "Czechia"/"Czech Republic"; "Ivory Coast"/"Côte d'Ivoire"). Build `pipeline/src/team_names.py` with one canonical name per team and explicit mapping dicts per source. A pytest must assert all 48 qualified teams resolve from every source. Unmapped names = hard pipeline failure (no silent drops).

Raw downloads cached to `pipeline/data/raw/` (gitignored), with ETag/If-Modified-Since where supported.

---

## 4. Repository structure

```
worldcup-2026-forecast/
├── PLAN.md                      # this file (gitignore it? NO — keep, it documents design)
├── projectsummary.md            # local only, GITIGNORED (Ian's rule)
├── README.md                    # 13-section standard (§13)
├── .gitignore                   # .venv, data/raw, mlruns, .env, projectsummary.md, node_modules
├── .env.example                 # FOOTBALL_DATA_TOKEN=
├── .github/workflows/daily.yml  # §12
├── run_pipeline.py              # CLI entrypoint: python run_pipeline.py [--skip-train] [--sims 10000]
├── requirements.txt
├── pipeline/
│   ├── src/
│   │   ├── config.py            # paths, constants, RNG seed policy, tournament dates
│   │   ├── team_names.py        # canonical names + per-source mappings
│   │   ├── ingest.py            # all three sources → data/raw, parsed to DataFrames
│   │   ├── elo.py               # World Football Elo engine (§5)
│   │   ├── features.py          # feature matrix builder (§6)
│   │   ├── train.py             # XGBoost training + evaluation (§7)
│   │   ├── poisson.py           # expected-goals + scoreline matrices (§7.4)
│   │   ├── knockout.py          # draw-reallocation for KO matches (§7.5)
│   │   ├── bracket.py           # 2026 bracket constants + third-place allocation (§8)
│   │   ├── simulate.py          # Monte Carlo engine (§8)
│   │   ├── freeze.py            # prediction freezing + scoring (§9)
│   │   └── publish.py           # JSON emitters (§10)
│   ├── data/
│   │   ├── raw/                 # gitignored
│   │   ├── processed/           # gitignored
│   │   └── frozen/              # COMMITTED: predictions_log.csv, scores.csv
│   └── models/                  # gitignored (retrained every run; reproducible)
├── tests/                       # pytest (§11)
└── web/                         # React + Vite + Tailwind v4 (§12... see §11 frontend)
    ├── public/data/             # COMMITTED: pipeline output JSONs
    └── src/ ...
```

---

## 5. Elo engine (`elo.py`) — exact specification

Compute our own Elo from the full 1872+ history (reproducible, no scraping dependency). Standard World Football Elo formula:

- Start every team at **1500**.
- Expected result: `We = 1 / (1 + 10 ** (-d / 400))` where `d = elo_home − elo_away + H`; **H = 100** home advantage, **H = 0** when `neutral == True`.
- Result `W`: win 1, draw 0.5, loss 0.
- K by match importance (map from martj42 `tournament` column — build an explicit mapping dict, default 30, friendlies 20):
  - World Cup finals: **60**
  - Continental finals (Euro, Copa América, AFCON, Asian Cup, Gold Cup) & intercontinental (Confederations Cup, Nations League finals): **50**
  - World Cup & continental qualifiers, UEFA Nations League: **40**
  - Other tournaments: **30**; Friendlies: **20**
- Goal-difference multiplier G: GD ≤ 1 → 1.0; GD = 2 → 1.5; GD ≥ 3 → `(11 + GD) / 8`.
- Update: `elo_new = elo_old + K * G * (W − We)` (zero-sum between the two teams).

Output: a function returning each team's Elo **as of any date** (needed for leakage-safe features), implemented as a chronological pass storing pre-match Elo on every match row. Cross-check vs eloratings.net per §3.

**2026 host note:** USA, Mexico, Canada get `neutral=False` home advantage in their own venues. martj42 has a `country` column — home advantage applies when `home_team == country` (the dataset's `neutral` flag already encodes this; trust the flag).

---

## 6. Feature engineering (`features.py`)

One row per match, **all features computed strictly from matches dated before the match** (walk-forward; this is the #1 leakage risk — see tests §11).

Features (home/away symmetric where applicable):
1. `elo_home`, `elo_away`, `elo_diff` (pre-match, incl. home advantage adjustment as separate flag, not baked into elo_diff)
2. `neutral` (bool), `is_home_host` (team playing in own country)
3. Rolling form, windows **5 and 10** matches: win rate, draw rate, goals scored avg, goals conceded avg → 8 features × 2 teams (Ian's "last ten matches" requirement)
4. Form quality: mean opponent Elo over last 10 (beats weak-opposition padding)
5. `rest_days_home/away` (days since previous international, capped at 60)
6. Head-to-head over last 10 meetings: home-team win rate, mean goal diff (0.5 / 0 when no history)
7. `importance` ordinal (friendly=0, other=1, qualifier=2, continental=3, world_cup=4)
8. `elo_trend_home/away`: Elo change over the trailing 365 days

Target: `outcome ∈ {home_win, draw, away_win}`.
Training rows: matches from **1990-01-01 onward** (Elo warm-started from 1872). Exponential time-decay sample weights, **half-life 4 years**, computed relative to each training cutoff.
Persist the exact feature column list to `models/feature_list.json` (inference reproducibility — cheatsheet checklist).

---

## 7. Models

### 7.1 XGBoost W/D/L classifier
`XGBClassifier(objective="multi:softprob", num_class=3)`. Tune with **Optuna, 75 trials**, objective = mean log loss over an expanding-window time series CV (3 folds: train≤2017/val 2018-19, train≤2019/val 2020-21, train≤2021/val 2022-23). `random_state=42` everywhere. Track in **MLflow** (local `mlruns/`, gitignored).

### 7.2 Validation protocol (report all of this in README)
- Hold-out test: **2024-01-01 → 2026-06-10** (never touched until final eval).
- Metrics: multiclass log loss, Brier score, accuracy, accuracy-of-favorite; **calibration curve** (10 bins) saved as JSON for the UI.
- Baselines (model must beat both on log loss): (a) Elo-only logistic regression on `elo_diff`; (b) historical W/D/L base rates.
- If softprob calibration is visibly poor, wrap with isotonic calibration fitted on 2022–2023 only.
- After sign-off, **retrain on all data through yesterday** for production; daily cron retrains daily (cheap at ~50k rows) — frozen predictions (§9) preserve honesty.

### 7.3 SHAP
`TreeExplainer`. For every upcoming fixture, persist top-6 signed SHAP contributions (toward the favorite's win class) into `match_detail` JSON — feeds the UI's "why" panel.

### 7.4 Poisson scoreline layer (`poisson.py`)
Two `XGBRegressor(objective="count:poisson")` models on the same features predicting `home_goals` and `away_goals` → λ_home, λ_away per fixture. Scoreline matrix: independent Poisson PMFs over 0–10 goals each (clip λ to [0.2, 4.5]). Outputs: most-likely scoreline, top-5 scorelines with probabilities, and a goal-difference distribution (used for simulated group tiebreakers). The classifier's W/D/L probs remain authoritative; **rescale the scoreline matrix's W/D/L mass to match the classifier's probabilities** (per-region renormalization) so the two never contradict in the UI.

### 7.5 Knockout draws (`knockout.py`)
No draws in KO rounds. `P(advance_home) = P(home_win) + P(draw) × We_shootout` where `We_shootout = 1/(1+10**(-elo_diff/400))` using neutral-venue Elo diff (mild favorite edge in ET/pens; defensible and simple). Document in model card.

---

## 8. Monte Carlo simulation (`simulate.py`, `bracket.py`)

**10,000 simulations**, NumPy vectorized where practical, seeded `seed = int(run_date.strftime("%Y%m%d"))` (reproducible per day). Completed matches always use actual results.

Per simulation:
1. **Group stage:** sample each remaining match's outcome from classifier probs; sample a scoreline from the Poisson matrix conditional on that outcome (for GD/GF tiebreakers).
2. **Standings per FIFA tiebreakers:** points → GD → GF → head-to-head (points, then GD, GF among tied teams) → random (we can't model fair-play points; note in model card).
3. **Third-place ranking** (across the 12 groups): points → GD → GF → random. Top 8 advance.
4. **Bracket — hard-code in `bracket.py` (verified vs Wikipedia 2026-06-12):**
   - M73: 2A v 2B M74: 1E v 3rd(A/B/C/D/F) M75: 1F v 2C M76: 1C v 2F
   - M77: 1I v 3rd(C/D/F/G/H) M78: 2E v 2I M79: 1A v 3rd(C/E/F/H/I) M80: 1L v 3rd(E/H/I/J/K)
   - M81: 1D v 3rd(B/E/F/I/J) M82: 1G v 3rd(A/E/H/I/J) M83: 2K v 2L M84: 1H v 2J
   - M85: 1B v 3rd(E/F/G/I/J) M86: 1J v 2H M87: 1K v 3rd(D/E/I/J/L) M88: 2D v 2G
   - **Third-place slot assignment:** FIFA's official table covers 495 combinations; do NOT hard-code it. Implement **constrained bipartite matching**: 8 qualified thirds × 8 slots, edge iff third's group ∈ slot's allowed set; find a perfect matching (backtracking over thirds in ranking order, preferring lowest-numbered available slot). Test: matching must exist for every one of the 495 C(12,8) combinations (exhaustive pytest over all combos — **VERIFY each combination yields a perfect matching; if any fails, the allowed-set constants are wrong, re-check Wikipedia**).
   - **R16→Final progression (M89–M104): NOT captured yet — extract the full winners-of-which-matches tree from the same Wikipedia knockout-stage page and hard-code as constants with a structure test (each of M89–M104 references exactly two earlier matches, every match feeds exactly one later match, finale = M104, third-place playoff = M103).**
5. KO matches resolved with §7.5 advance probabilities.

Aggregate outputs: per-team P(reach R32/R16/QF/SF/Final/win); per-group finish-position distribution + P(qualify); per-match outcome frequencies; champion odds ranking. Sanity invariants (asserted in pipeline AND tests): champion probs sum to 1.0 ± 1e-6; exactly 12 group winners per sim; P(reach R32) ≥ P(reach R16) ≥ … per team.

---

## 9. Prediction freezing & self-scoring (`freeze.py`)

- On every run: for each fixture kicking off within **72h** that has no frozen entry yet, append to `pipeline/data/frozen/predictions_log.csv`: `frozen_at_utc, match_id, kickoff_utc, stage, home, away, p_home, p_draw, p_away, pred_score_home, pred_score_away, model_version (git sha)`. **Append-only — never rewrite history.** Committed by the Action = public, timestamped, auditable.
- Scoring: for frozen entries whose result is now known → `scores.csv` with realized outcome, per-match Brier (multiclass) and log loss, favorite-correct flag, exact-score-correct flag. Emit cumulative metrics + calibration bins to `accuracy.json`.

---

## 10. JSON schema contracts (`publish.py` → `web/public/data/`)

The frontend builds against these exact schemas; pipeline tests validate every emitted file against them (use `jsonschema` lib, schemas in `pipeline/schemas/`). All files carry `{"generated_at_utc": ..., "model_version": ...}` envelope.

1. `meta.json` — run timestamp, git sha, n_sims, training rows, test metrics summary.
2. `champion_odds.json` — `[{team, code, p_champion, p_final, p_sf, p_qf, p_r16, p_r32, delta_vs_yesterday}]` sorted desc. (Compute delta by reading the previous file before overwrite.)
3. `groups.json` — 12 groups: current standings (from football-data.org) + per-team `{p_win_group, p_runner_up, p_third_advance, p_eliminated}`.
4. `matches.json` — every match (played + upcoming): id, stage, group, kickoff_utc, venue/city if available, teams, status, actual score or `{p_home,p_draw,p_away}`, most-likely score.
5. `match_detail/{match_id}.json` — probs, top-5 scorelines, scoreline matrix (11×11), SHAP top-6 factors with plain-English labels, both teams' last-10 form strips (W/D/L), h2h summary, Elo history (trailing 2y, monthly).
6. `bracket.json` — R32→Final tree; for each slot, the probability distribution over teams that could occupy it (top 6 + "other").
7. `accuracy.json` — cumulative Brier/log-loss/favorite-accuracy over time (array per matchday), calibration bins, n_scored, and the full frozen-prediction history for the "receipts" table.

---

## 11. Frontend (`web/`) — CupCast 2026

**Stack:** React 18 + Vite + Tailwind **v4** (REMEMBER the v4 cascade bug from the portfolio build — check `reference_portfolio_build.md` memory), Recharts for charts, framer-motion for animation. No backend, no router needed beyond `react-router-dom` if multi-page (use it; 5 routes). Fetch JSONs from `/data/` at load with a thin `useData` hook + skeleton loaders.

**Design brief (Ian's preferences: animated, dynamic, dark):** dark theme, one bold accent (suggest electric green `#00E676` on near-black, World-Cup-grass vibe), Inter or Space Grotesk, staggered card entrances, animated probability bars that fill on scroll, page-enter transitions, count-up numbers. Apply the `/frontend-design` skill when building. Mobile-first — this is a share-with-friends site.

**Pages:**
1. **Home** — hero with next matches (probability bars, countdown), champion-odds top-10 horizontal bar chart with daily delta arrows, headline stat cards (favorite, biggest mover, model accuracy so far).
2. **Bracket** — interactive R32→Final tree; each slot shows most-likely team + probability, click → distribution popover. (Hardest component — build last; degrade gracefully on mobile to vertical accordion.)
3. **Groups** — 12 group cards: live table + animated qualification probability bars per team.
4. **Match detail** (route per match) — probability donut, top-5 scorelines, SHAP "why the model thinks so" panel (signed factor bars with plain-English labels), form strips, h2h, Elo trend chart.
5. **Model performance** — the receipts page: calibration curve, Brier trend vs matchday, frozen-predictions table (what we said / what happened), honest model-card prose (~55-60% ceiling, what the model can't see: injuries, lineups, weather).

Footer: data attributions (martj42, eloratings.net, football-data.org), "Built by Ian Mwendwa" + portfolio/GitHub links. No AI mentions anywhere (Ian's rule).

---

## 12. Automation & deployment

**GitHub Actions** `.github/workflows/daily.yml`:
- `schedule: cron "0 9 * * *"` (09:00 UTC — last North-American kickoff ends ~04:00–05:00 UTC; martj42 updates daily) + `workflow_dispatch` for manual runs.
- Steps: checkout (full depth 1 ok) → setup Python 3.12 + uv → `uv pip install -r requirements.txt --system` → `python run_pipeline.py` (env `FOOTBALL_DATA_TOKEN` from repo secret) → `git add web/public/data pipeline/data/frozen && git commit -m "Daily forecast update YYYY-MM-DD" || echo "no changes"` → push. Concurrency group to prevent overlap; ~10 min budget.
- Commit author: Ian's GitHub identity (configure `user.name`/`user.email` in workflow). **No AI co-author lines** (Ian's rule applies to automated commits too).
- **Vercel:** project rooted at `web/`, auto-deploy on push to main. `vercel.json` with `cleanUrls` and cache headers for `/data/*.json` (`max-age=300`).

**requirements.txt** (CVE-safe floors per memory; run the security-audit workflow — install first, then `pip-audit --local` — before first push):
`xgboost>=2.0.0, scikit-learn>=1.5.0, shap>=0.45.0, optuna>=3.6.0, mlflow>=2.15.0, pandas>=2.2.0, numpy>=1.26.0, requests>=2.33.0, python-dotenv>=1.2.2, joblib>=1.4.0, jsonschema>=4.21.0, pytest>=9.0.3, pip-audit`

---

## 13. Tests, docs, process (Ian's standing rules — all apply)

**pytest targets (~50 tests):**
- Elo: hand-computed known cases (incl. GD multiplier branches, K mapping, home adv, zero-sum), convergence sanity (Brazil/Argentina/France top-tier by 2026).
- Leakage: for sampled matches, recompute every feature using only prior matches and assert equality with the pipeline's matrix; assert no feature column correlates with target via future info (rebuild row with the match's own result removed → identical features).
- Team names: all 48 teams resolve from all 3 sources; unmapped → raises.
- Bracket: exhaustive 495-combination matching test; progression-tree structure test; tiebreaker unit tests (constructed h2h scenarios).
- Simulation invariants (§8) on a 200-sim smoke run.
- Schema validation of every emitted JSON.
- Freeze: append-only behavior, no re-freezing, scoring math (hand-computed Brier).

**Process:** `uv venv` + `.venv`; build in phases 0–9 matching §4 module order, **validate each phase before the next** (Phase 3 gate: report test-set log loss vs both baselines to Ian before proceeding); test locally before any push; update `projectsummary.md` + README incrementally with exact numbers as milestones land; screenshots → `assets/`; README in the 13-section standard (NSE/Forex structure) including Key Design Decisions (self-computed Elo, draw-reallocation formula, matching-vs-lookup for thirds, static-JSON architecture) and the model card; full README accuracy audit before push; pip-audit before push; no AI credits in commits.

**Ian's action items (blocking):**
1. Register at football-data.org → free API token → `.env` locally + `FOOTBALL_DATA_TOKEN` repo secret.
2. Create GitHub repo `World-Cup-2026-Forecast`, link Vercel project to `web/`.
3. Allow GitHub Actions to push (default GITHUB_TOKEN write permission in workflow `permissions: contents: write`).

**Risks & mitigations:** martj42 lag on same-day results → football-data.org results are authoritative for WC 2026 matches (merge both, prefer official). football-data.org payload surprises → VERIFY shape first, parser behind one module. Group-stage clock → ship Home+Groups pages with v1, Bracket page can follow 1–2 days later. Draw rate over-prediction in KO sims → covered by §7.5. Cold start with zero scored predictions → accuracy page shows "first matchday pending" empty state.
