"""CupCast 2026 — daily pipeline orchestrator.

    ingest -> elo -> features -> (train) -> simulate -> publish -> freeze -> score

Usage:
    python run_pipeline.py                 # full run, retrain, 10k sims
    python run_pipeline.py --skip-train    # reuse existing models
    python run_pipeline.py --sims 2000     # fewer sims (dev)
    python run_pipeline.py --trials 75     # Optuna trials when training
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "pipeline" / "src"
sys.path.insert(0, str(SRC))

import config as C          # noqa: E402
import artifacts            # noqa: E402
import ingest               # noqa: E402
import simulate             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--refit", action="store_true",
                    help="refit on fresh data with committed params (daily CI path)")
    ap.add_argument("--sims", type=int, default=C.N_SIMULATIONS)
    ap.add_argument("--trials", type=int, default=75)
    ap.add_argument("--no-refresh", action="store_true", help="use cached raw data")
    args = ap.parse_args()

    t0 = time.time()
    print("[1/6] Ingesting + building artifacts (Elo, features) ...")
    art = artifacts.build(refresh=not args.no_refresh)
    print(f"      history={len(art['history']):,}  train_rows={len(art['train']):,}")

    if args.skip_train and (C.MODELS / "wdl_model.joblib").exists():
        import json
        metrics = json.loads((C.MODELS / "metrics.json").read_text())
        print("[2/6] Skipping training (reusing saved models).")
    elif args.refit:
        print("[2/6] Refitting on fresh data with committed params (no tuning) ...")
        import train
        metrics = train.refit(art)
        train.print_gate_report(metrics)
    else:
        print(f"[2/6] Training W/D/L + Poisson ({args.trials} Optuna trials) ...")
        import train
        metrics = train.train_all(art, n_trials=args.trials)
        train.print_gate_report(metrics)

    print(f"[3/6] Simulating tournament ({args.sims:,} runs) ...")
    fixtures = ingest.load_fixtures(refresh=not args.no_refresh)
    sim = simulate.Simulator(art, fixtures, n_sims=args.sims, seed=C.run_seed())
    sim_out = sim.run()

    print("[4/6] Scoring resolved predictions ...")
    import freeze
    n_scored = freeze.score_resolved(fixtures)
    print(f"      newly scored: {n_scored}")

    print("[5/6] Publishing JSON contracts ...")
    import publish
    version = publish.git_sha()
    pub = publish.Publisher(art, fixtures, sim_out, metrics, version)
    predictions = pub.run()

    print("[6/6] Freezing due predictions (<=72h to kickoff) ...")
    n_frozen = freeze.freeze_due(fixtures, predictions, version)
    print(f"      newly frozen: {n_frozen}")

    champ = sorted(sim_out["teams"].values(), key=lambda d: -d["p_champion"])[:5]
    print("\nTop 5 champion odds:")
    for d in champ:
        print(f"  {d['team']:<16} {d['p_champion']*100:5.1f}%")
    print(f"\nDone in {time.time()-t0:.0f}s. Outputs -> web/public/data/")


if __name__ == "__main__":
    main()
