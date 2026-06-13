# CupCast 2026 — World Cup Machine-Learning Forecast

A daily-updating machine-learning forecast for the 2026 FIFA World Cup. The model rates all
48 teams with a self-computed Elo system, predicts every remaining fixture with a gradient-boosted
classifier, projects scorelines with a Poisson goal model, and simulates the whole tournament
10,000 times to produce champion odds and per-team advancement probabilities. Every prediction is
**frozen before kickoff and graded in the open**, so the model's accuracy is auditable rather than
asserted.

**Live site:** _deployed on Vercel_ · **Stack:** Python (XGBoost · scikit-learn · SHAP · Optuna) → static JSON → React + Vite + Tailwind v4

![Home](assets/home.png)

| Groups | Bracket | Match detail |
|---|---|---|
| ![Groups](assets/groups.png) | ![Bracket](assets/bracket.png) | ![Match detail](assets/match-detail.png) |

---

## 1. Overview

The 2026 World Cup (11 June – 19 July) is the first 48-team edition: 12 groups of four, a new
round of 32 (the top two from each group plus the eight best third-placed teams), and 104 matches.
CupCast forecasts it end to end:

- **Match probabilities** — win / draw / loss for every fixture, with a most-likely scoreline.
- **Tournament simulation** — 10,000 Monte Carlo runs → champion odds, group-qualification odds,
  and round-by-round advancement probabilities for all 48 teams.
- **Self-scoring** — predictions are committed to an append-only log before kickoff and graded
  (Brier score, log loss, calibration) as results arrive.

The forecast refreshes daily via GitHub Actions and redeploys automatically — no servers, no paid
APIs, real data only.

## 2. Results

Held-out test set (2,543 men's internationals, Jan 2024 – Jun 2026, never used for tuning):

| Metric | CupCast XGBoost | Elo-only baseline | Base-rate baseline |
|---|---|---|---|
| Log loss (lower better) | **0.8601** | 0.8676 | 1.0537 |
| Favourite accuracy | **60.0%** | — | — |
| Multiclass Brier | **0.5056** | — | — |

The model beats both baselines on log loss. The margin over Elo is deliberately modest and honest —
Elo is a very strong predictor in international football, and match-level prediction tops out near
55–60% accuracy for anyone, bookmakers included. The value is in *calibrated probabilities* and the
tournament simulation, not in pretending to call upsets.

Champion odds at build time (10,000 simulations) lined up closely with live betting markets:

| Team | Champion | Reach final | Reach semis |
|---|---|---|---|
| Spain | 27.7% | 40% | 52% |
| Argentina | 18.6% | 31% | 44% |
| France | 10.3% | 19% | 35% |
| England | 7.8% | 15% | 28% |
| Brazil | 5.7% | 12% | 24% |

## 3. Architecture

```
GitHub Actions (daily 09:00 UTC)
  └─ run_pipeline.py
       ingest → elo → features → refit → simulate → publish → freeze → score
  └─ commit web/public/data/*.json + frozen predictions → Vercel auto-deploys

React + Vite + Tailwind v4 (static)  ←  reads /data/*.json
```

The pipeline (`pipeline/`) and the frontend (`web/`) communicate only through seven versioned JSON
contracts, each validated against a JSON Schema before it is written.

## 4. Data sources (all free, verified at build)

| Source | Used for | Access |
|---|---|---|
| [martj42 international results](https://github.com/martj42/international_results) | 49,410-match training history (1872→present, updated daily) | raw CSV |
| [eloratings.net](https://eloratings.net/) | cross-check for the self-computed Elo | public TSV |
| [football-data.org](https://www.football-data.org/) | official WC 2026 fixtures, results, standings | free tier (token) |

Team names differ across sources (Czechia / Czech Republic, Congo DR / DR Congo, Cape Verde Islands /
Cape Verde, Curaçao's cedilla). A single canonical registry maps every source, and a test asserts all
48 qualified teams resolve from every source — an unmapped name is a hard failure, never a silent drop.

## 5. The model

- **Elo engine** — World Football Elo computed from the full 1872+ history in one chronological pass
  (K by match importance: 60 for World Cup down to 20 for friendlies; goal-difference multiplier;
  +100 home advantage, zero on neutral ground). Self-computed so it is fully reproducible and
  available *as of any date* for leakage-safe features.
- **Features (30)** — Elo and Elo gap, recent form over the last 5 and 10 matches (win/draw rates,
  goals for/against), strength of schedule, rest days, head-to-head record, 1-year Elo momentum, and
  match importance. Every feature is computed walk-forward using only matches *before* the one being
  predicted; a unit test rebuilds each row from a truncated history to prove no future information leaks.
- **Classifier** — XGBoost `multi:softprob`, tuned with Optuna (expanding-window time-series CV),
  trained on 32,292 matches from 1990 onward with a 4-year exponential time-decay weighting.
- **Poisson scoreline model** — two `count:poisson` regressors give expected goals; the resulting
  scoreline matrix is renormalised so its win/draw/loss mass matches the classifier (which stays
  authoritative). Scorelines are sampled from this matrix in simulation, keeping points and goal
  difference mutually consistent.
- **Explainability** — SHAP `TreeExplainer` surfaces the top factors behind each fixture's prediction
  in plain English on the match page.

## 6. Tournament simulation

Each of the 10,000 simulations plays out every remaining match, builds group tables with FIFA
tiebreakers (points → goal difference → goals for → head-to-head → random), ranks the eight best
third-placed teams, resolves the round-of-32 bracket, and plays the knockout tree to a champion.
Completed matches always use their real results.

The round-of-32 third-place allocation (FIFA publishes a 495-row lookup table) is solved as a
constrained bipartite matching instead of hard-coding the table — and a test verifies a valid
assignment exists for **all 495** of the C(12,8) combinations. Knockout draws are resolved by sending
extra-time/penalties to the neutral-venue Elo favourite. Invariants (champion probabilities sum to 1,
exactly 12 group winners per simulation, monotonic advancement) are asserted in both the pipeline and
the test suite.

## 7. Self-scoring (the receipts)

Within 72 hours of kickoff, each fixture's prediction is appended — once, append-only — to
`pipeline/data/frozen/predictions_log.csv`, which the daily Action commits. This is a timestamped,
public record that the forecast was made *before* the match. As results land, frozen predictions are
graded (Brier, log loss, favourite-correct, exact-score-correct) and the Model page shows the running
calibration curve, accuracy trend, and a "what we said / what happened" receipts table.

## 8. Project structure

```
worldcup-2026-forecast/
├── run_pipeline.py              # orchestrator: ingest→…→score
├── requirements.txt
├── pipeline/
│   ├── src/
│   │   ├── config.py            # paths, constants, RNG policy
│   │   ├── team_names.py        # canonical registry + 12 groups
│   │   ├── ingest.py            # martj42 + football-data.org
│   │   ├── elo.py               # World Football Elo engine
│   │   ├── features.py          # walk-forward feature matrix
│   │   ├── train.py             # XGBoost + Optuna + Poisson + refit
│   │   ├── predict.py           # inference + scoreline matrices
│   │   ├── bracket.py           # R32→Final structure + 3rd-place matching
│   │   ├── knockout.py          # extra-time/penalties resolution
│   │   ├── simulate.py          # 10k Monte Carlo engine
│   │   ├── freeze.py            # prediction freezing + scoring
│   │   └── publish.py           # seven JSON contracts (+ SHAP)
│   ├── schemas/schemas.py       # JSON Schemas for every contract
│   ├── data/frozen/             # committed: predictions_log.csv, scores.csv
│   └── best_params.json         # committed tuned params (daily refit)
├── tests/                       # 44 pytest tests
├── web/                         # React + Vite + Tailwind v4 (CupCast 2026)
│   ├── src/pages/               # Home · Groups · Bracket · Match · Model
│   └── public/data/             # committed forecast JSON (consumed by UI)
└── .github/workflows/daily.yml  # daily refit + redeploy
```

## 9. Setup

```bash
# Pipeline
uv venv && uv pip install -r requirements.txt
cp .env.example .env            # add your free football-data.org token

# Frontend
cd web && npm install
```

## 10. Usage

```bash
# Full run: tune + train + 10k sims + publish (first time)
python run_pipeline.py --trials 75

# Daily path: refit on fresh data with committed params, 10k sims
python run_pipeline.py --refit --sims 10000

# Fast dev iteration: reuse models, fewer sims, cached raw data
python run_pipeline.py --skip-train --sims 1000 --no-refresh

# Frontend
cd web && npm run dev            # local dev
npm run build                    # production build → web/dist
```

## 11. Testing

```bash
python -m pytest tests/ -q       # 44 tests
```

Coverage includes hand-computed Elo cases, the no-future-leakage proof, team-name resolution across
all sources, the exhaustive 495-combination bracket-matching guarantee, simulation invariants, the
freeze/score math, and JSON-Schema validation of every published file.

## 12. Key design decisions

- **Self-computed Elo, not scraped.** Reproducible, dependency-free, and queryable as of any date —
  essential for leakage-safe features. eloratings.net is used only as a sanity cross-check.
- **Static JSON, no backend.** Predictions change a few times a day, so a daily batch that commits
  JSON and lets Vercel serve it statically is cheaper, faster, and simpler than any live server.
- **Classifier authoritative, Poisson for texture.** The W/D/L classifier sets outcome probabilities;
  the Poisson matrix is renormalised to agree with it, so scorelines never contradict the headline odds.
- **Bipartite matching over a 495-row lookup.** Computing the third-place allocation (and testing all
  495 cases) is more robust than transcribing FIFA's table by hand.
- **Tune once, refit daily.** Hyperparameters are tuned locally and committed; CI refits on fresh data
  without re-running Optuna, keeping daily runs fast and the model stable.
- **Freeze before kickoff.** Honesty is enforced structurally — the append-only log means accuracy
  can be audited, not just claimed.

## 13. Skills demonstrated

- **ML engineering** — gradient boosting, Poisson regression, Optuna tuning, time-series cross-validation,
  SHAP explainability, calibration, and rigorous leakage control.
- **Simulation** — a 10,000-run Monte Carlo tournament engine with real FIFA rules and constrained
  bracket matching.
- **Data engineering** — multi-source ingestion, canonical entity resolution, schema-validated data
  contracts, and a scheduled, self-committing pipeline.
- **Frontend** — a polished, animated React + Tailwind v4 interface with interactive bracket,
  probability visualisations, and an honest model-performance page.
- **Software practice** — 44 tests, reproducible seeds, security-audited dependencies, and CI/CD to Vercel.

---

*Forecasts are probabilistic and for interest only. The model cannot see injuries, suspensions,
lineups, or weather — treat every number as a probability, not a promise.*

Data: martj42 · eloratings.net · football-data.org. Built by Ian Mwendwa.
