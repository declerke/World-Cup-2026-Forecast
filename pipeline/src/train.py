"""Train and validate the W/D/L classifier + Poisson scoreline models.

Protocol (see PLAN.md §7):
  - Optuna tuning, objective = mean log loss over expanding-window time-series CV.
  - Held-out test 2024-01-01 .. 2026-06-10, never tuned on.
  - Must beat two baselines on log loss: Elo-only logistic + historical base rates.
  - Production model retrains on ALL data; honesty preserved by frozen predictions.
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from xgboost import XGBClassifier, XGBRegressor

import config as C
from features import FEATURE_COLUMNS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

LABELS = ["home_win", "draw", "away_win"]
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}

# Tuned once locally, committed (models/ is gitignored). CI refits on fresh data
# daily using these params instead of re-running Optuna — stable + fast.
BEST_PARAMS_PATH = C.PIPELINE / "best_params.json"

CV_FOLDS = [  # (train_end, val_start, val_end)
    ("2017-12-31", "2018-01-01", "2019-12-31"),
    ("2019-12-31", "2020-01-01", "2021-12-31"),
    ("2021-12-31", "2022-01-01", "2023-12-31"),
]


def _decay_weights(dates: pd.Series, ref: pd.Timestamp) -> np.ndarray:
    age_years = (ref - dates).dt.days / 365.25
    return np.power(0.5, age_years / C.TIME_DECAY_HALF_LIFE_YEARS).to_numpy()


def _Xy(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["outcome"].map(LABEL_IDX).to_numpy()
    return X, y


def _objective(trial, train: pd.DataFrame) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
    }
    losses = []
    for tr_end, va_start, va_end in CV_FOLDS:
        tr = train[train["date"] <= tr_end]
        va = train[(train["date"] >= va_start) & (train["date"] <= va_end)]
        if len(va) == 0:
            continue
        Xtr, ytr = _Xy(tr)
        Xva, yva = _Xy(va)
        w = _decay_weights(tr["date"], tr["date"].max())
        model = XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            tree_method="hist", random_state=C.SEED, n_jobs=-1, **params,
        )
        model.fit(Xtr, ytr, sample_weight=w)
        proba = model.predict_proba(Xva)
        losses.append(log_loss(yva, proba, labels=[0, 1, 2]))
    return float(np.mean(losses))


def _baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    pre = train[train["date"] < C.TEST_FROM]
    _, ytr = _Xy(pre)
    Xte, yte = _Xy(test)

    # Base-rate baseline
    base = np.bincount(ytr, minlength=3) / len(ytr)
    base_proba = np.tile(base, (len(yte), 1))

    # Elo-only logistic regression on [elo_diff, neutral] (standardised to converge)
    cols = [FEATURE_COLUMNS.index("elo_diff"), FEATURE_COLUMNS.index("neutral")]
    Xtr_elo = pre[FEATURE_COLUMNS].to_numpy()[:, cols]
    mu, sigma = Xtr_elo.mean(axis=0), Xtr_elo.std(axis=0) + 1e-9
    lr = LogisticRegression(max_iter=2000)
    lr.fit((Xtr_elo - mu) / sigma, ytr)
    elo_proba = lr.predict_proba((Xte[:, cols] - mu) / sigma)

    return {
        "base_rate_logloss": float(log_loss(yte, base_proba, labels=[0, 1, 2])),
        "elo_only_logloss": float(log_loss(yte, elo_proba, labels=[0, 1, 2])),
    }


def _fit_production(train: pd.DataFrame, best: dict):
    """Refit production W/D/L + Poisson models on ALL data and evaluate on holdout."""
    test = train[(train["date"] >= C.TEST_FROM) & (train["date"] <= C.TEST_TO)].copy()
    dev = train[train["date"] < C.TEST_FROM].copy()

    Xdev, ydev = _Xy(dev)
    Xte, yte = _Xy(test)
    wdev = _decay_weights(dev["date"], dev["date"].max())
    eval_model = XGBClassifier(objective="multi:softprob", num_class=3,
                               eval_metric="mlogloss", tree_method="hist",
                               random_state=C.SEED, n_jobs=-1, **best)
    eval_model.fit(Xdev, ydev, sample_weight=wdev)
    proba_te = eval_model.predict_proba(Xte)
    metrics = {
        "test_logloss": float(log_loss(yte, proba_te, labels=[0, 1, 2])),
        "test_brier": _multiclass_brier(yte, proba_te),
        "test_accuracy": float(accuracy_score(yte, proba_te.argmax(axis=1))),
        "n_test": int(len(test)), "n_dev": int(len(dev)),
        "best_params": best, "calibration": _calibration(yte, proba_te),
    }
    metrics.update(_baselines(train, test))
    metrics["beats_elo_only"] = metrics["test_logloss"] < metrics["elo_only_logloss"]
    metrics["beats_base_rate"] = metrics["test_logloss"] < metrics["base_rate_logloss"]

    Xall, yall = _Xy(train)
    wall = _decay_weights(train["date"], train["date"].max())
    prod = XGBClassifier(objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", tree_method="hist",
                         random_state=C.SEED, n_jobs=-1, **best)
    prod.fit(Xall, yall, sample_weight=wall)

    pois_params = dict(objective="count:poisson", n_estimators=300, learning_rate=0.05,
                       max_depth=5, subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", random_state=C.SEED, n_jobs=-1)
    pois_home = XGBRegressor(**pois_params)
    pois_away = XGBRegressor(**pois_params)
    pois_home.fit(Xall, train["home_goals"].to_numpy(), sample_weight=wall)
    pois_away.fit(Xall, train["away_goals"].to_numpy(), sample_weight=wall)

    joblib.dump(prod, C.MODELS / "wdl_model.joblib")
    joblib.dump(pois_home, C.MODELS / "poisson_home.joblib")
    joblib.dump(pois_away, C.MODELS / "poisson_away.joblib")
    (C.MODELS / "feature_list.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (C.MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def refit(art: dict) -> dict:
    """Daily CI path: refit on fresh data using committed tuned params (no Optuna)."""
    if not BEST_PARAMS_PATH.exists():
        raise RuntimeError("best_params.json missing — run a full `train_all` first.")
    best = json.loads(BEST_PARAMS_PATH.read_text())
    return _fit_production(art["train"].copy(), best)


def _multiclass_brier(y, proba) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _calibration(y, proba, n_bins=10) -> list[dict]:
    """Reliability of the predicted favourite probability."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if m.sum() > 0:
            out.append({"bin": round((bins[i] + bins[i + 1]) / 2, 3),
                        "predicted": round(float(conf[m].mean()), 4),
                        "actual": round(float(correct[m].mean()), 4),
                        "count": int(m.sum())})
    return out


def train_all(art: dict, n_trials: int = 75) -> dict:
    train = art["train"].copy()
    test = train[(train["date"] >= C.TEST_FROM) & (train["date"] <= C.TEST_TO)].copy()
    dev = train[train["date"] < C.TEST_FROM].copy()

    # ---- tune on dev only ----
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(lambda t: _objective(t, dev), n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))   # committed for daily refit

    # ---- evaluate tuned config on held-out test ----
    Xdev, ydev = _Xy(dev)
    Xte, yte = _Xy(test)
    wdev = _decay_weights(dev["date"], dev["date"].max())
    eval_model = XGBClassifier(objective="multi:softprob", num_class=3,
                               eval_metric="mlogloss", tree_method="hist",
                               random_state=C.SEED, n_jobs=-1, **best)
    eval_model.fit(Xdev, ydev, sample_weight=wdev)
    proba_te = eval_model.predict_proba(Xte)
    metrics = {
        "test_logloss": float(log_loss(yte, proba_te, labels=[0, 1, 2])),
        "test_brier": _multiclass_brier(yte, proba_te),
        "test_accuracy": float(accuracy_score(yte, proba_te.argmax(axis=1))),
        "n_test": int(len(test)), "n_dev": int(len(dev)),
        "best_params": best, "best_cv_logloss": float(study.best_value),
        "calibration": _calibration(yte, proba_te),
    }
    metrics.update(_baselines(train, test))
    metrics["beats_elo_only"] = metrics["test_logloss"] < metrics["elo_only_logloss"]
    metrics["beats_base_rate"] = metrics["test_logloss"] < metrics["base_rate_logloss"]

    # ---- production model: retrain on ALL data ----
    Xall, yall = _Xy(train)
    wall = _decay_weights(train["date"], train["date"].max())
    prod = XGBClassifier(objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", tree_method="hist",
                         random_state=C.SEED, n_jobs=-1, **best)
    prod.fit(Xall, yall, sample_weight=wall)

    # ---- Poisson goal models (count:poisson) ----
    pois_params = dict(objective="count:poisson", n_estimators=300,
                       learning_rate=0.05, max_depth=5, subsample=0.8,
                       colsample_bytree=0.8, tree_method="hist",
                       random_state=C.SEED, n_jobs=-1)
    pois_home = XGBRegressor(**pois_params)
    pois_away = XGBRegressor(**pois_params)
    pois_home.fit(Xall, train["home_goals"].to_numpy(), sample_weight=wall)
    pois_away.fit(Xall, train["away_goals"].to_numpy(), sample_weight=wall)

    # ---- persist ----
    joblib.dump(prod, C.MODELS / "wdl_model.joblib")
    joblib.dump(pois_home, C.MODELS / "poisson_home.joblib")
    joblib.dump(pois_away, C.MODELS / "poisson_away.joblib")
    (C.MODELS / "feature_list.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (C.MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _mlflow_log(metrics, best)
    return metrics


def _mlflow_log(metrics: dict, params: dict) -> None:
    try:
        import os
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
        mlflow.set_tracking_uri((C.PIPELINE / "mlruns").as_uri())
        mlflow.set_experiment("cupcast_wdl")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
    except Exception as e:  # pragma: no cover
        print(f"(mlflow logging skipped: {e})")


def print_gate_report(m: dict) -> None:
    print("\n" + "=" * 60)
    print("  PHASE 3 VALIDATION GATE - held-out test 2024-01 .. 2026-06")
    print("=" * 60)
    print(f"  test rows           : {m['n_test']:,}")
    print(f"  XGBoost  log loss   : {m['test_logloss']:.4f}")
    print(f"  Elo-only log loss   : {m['elo_only_logloss']:.4f}   "
          f"{'BEAT' if m['beats_elo_only'] else 'LOST'}")
    print(f"  Base-rate log loss  : {m['base_rate_logloss']:.4f}   "
          f"{'BEAT' if m['beats_base_rate'] else 'LOST'}")
    print(f"  test Brier          : {m['test_brier']:.4f}")
    print(f"  test accuracy       : {m['test_accuracy']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    import artifacts
    art = artifacts.load()
    n = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 75
    metrics = train_all(art, n_trials=n)
    print_gate_report(metrics)
