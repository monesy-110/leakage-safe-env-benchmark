"""Equal-budget hyperparameter tuning + leakage-free evaluation of tuned models.

Phase 1 (default run): Optuna [Akiba et al. 2019], 15 trials (timeout 420 s)
per (dataset, model). Objective = primary metric on a FIXED random 80/20
validation split (seed 0) — i.e. tuning happens under the leaky protocol,
mirroring common practice, so we can later ask whether tuned-and-tested-honestly
models still differ. Best params saved incrementally to results/tuned_params.json.

Phase 2 (--evaluate): refit each tuned configuration under every applicable
protocol x seed (0-9) and append to results/tuned_results.csv (fixed-field
schema, same FIELDS as benchmark.py).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import root_mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

from benchmark import (ROOT, build_xy, make_split, make_preprocessor,  # noqa: E402
                       FIELDS)

RESULTS_DIR = ROOT / "results"
PARAMS_JSON = RESULTS_DIR / "tuned_params.json"
TUNED_CSV = RESULTS_DIR / "tuned_results.csv"
N_TRIALS = 15
TIMEOUT_S = 420
MAX_TRAIN_ROWS = {"knn": 60_000, "mlp": 120_000}


def sample_params(mname: str, task: str, trial) -> dict:
    if mname in ("elasticnet", "logreg"):
        if task == "regression":
            return dict(alpha=trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                        l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                        max_iter=20000)
        return dict(C=trial.suggest_float("C", 1e-3, 100.0, log=True),
                    max_iter=5000)
    if mname == "knn":
        return dict(n_neighbors=trial.suggest_int("n_neighbors", 3, 50),
                    weights=trial.suggest_categorical("weights",
                                                      ["uniform", "distance"]),
                    p=trial.suggest_int("p", 1, 2), n_jobs=-1)
    if mname == "rf":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 500,
                                                   step=100),
                    max_depth=trial.suggest_categorical(
                        "max_depth", [None, 8, 12, 16, 20, 24]),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    max_features=trial.suggest_categorical(
                        "max_features", ["sqrt", 0.5, 1.0]),
                    random_state=0, n_jobs=-1)
    if mname == "xgboost":
        p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 600,
                                                step=100),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3,
                                                   log=True),
                 max_depth=trial.suggest_int("max_depth", 3, 10),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree",
                                                      0.6, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0,
                                                log=True),
                 tree_method="hist", random_state=0, n_jobs=-1)
        if task == "classification":
            p["eval_metric"] = "logloss"
        return p
    if mname == "lightgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 600,
                                                   step=100),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3,
                                                      log=True),
                    num_leaves=trial.suggest_int("num_leaves", 15, 200),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree",
                                                         0.6, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0,
                                                   log=True),
                    random_state=0, n_jobs=-1, verbosity=-1)
    if mname == "mlp":
        return dict(hidden_layer_sizes=trial.suggest_categorical(
                        "hidden_layer_sizes",
                        [(64,), (128,), (64, 32), (128, 64), (256, 128)]),
                    alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                    learning_rate_init=trial.suggest_float(
                        "learning_rate_init", 1e-4, 1e-2, log=True),
                    early_stopping=True, max_iter=300, random_state=0)
    raise ValueError(mname)


def build(mname: str, task: str, params: dict):
    if mname == "elasticnet":
        return ElasticNet(**params)
    if mname == "logreg":
        return LogisticRegression(**params)
    if mname in ("xgboost", "lightgbm"):
        import xgboost, lightgbm
        if mname == "xgboost":
            return (xgboost.XGBRegressor(**params) if task == "regression"
                    else xgboost.XGBClassifier(**params))
        return (lightgbm.LGBMRegressor(**params) if task == "regression"
                else lightgbm.LGBMClassifier(**params))
    if task == "regression":
        return {"knn": KNeighborsRegressor, "rf": RandomForestRegressor,
                "mlp": MLPRegressor}[mname](**params)
    return {"knn": KNeighborsClassifier, "rf": RandomForestClassifier,
            "mlp": MLPClassifier}[mname](**params)


def tune_one(X, y, meta, mname, scale) -> dict | None:
    task = meta["task"]
    idx = np.arange(len(X))
    strat = y if task == "classification" else None
    tr, va = train_test_split(idx, test_size=0.2, random_state=0,
                              stratify=strat)
    Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
    direction = "minimize" if task == "regression" else "maximize"

    def objective(trial):
        params = sample_params(mname, task, trial)
        pipe = Pipeline([("prep", make_preprocessor(meta, scale)),
                         ("model", build(mname, task, params))])
        pipe.fit(Xtr, ytr)
        if task == "regression":
            return root_mean_squared_error(yva, pipe.predict(Xva))
        return roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction=direction)
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_S,
                   show_progress_bar=False)
    print(f"    {mname}: best={study.best_value:.4f} "
          f"({time.time() - t0:.0f}s, {len(study.trials)} trials)", flush=True)
    return study.best_params


def load_data(manifest: dict):
    for name, meta in manifest.items():
        df = pd.read_parquet(ROOT / meta["processed_file"])
        yield name, meta, *build_xy(df, meta)


def run_tuning() -> None:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    store = (json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
             if PARAMS_JSON.exists() else {})
    for name, meta, X, y in load_data(manifest):
        print(f"== tuning {name} ({meta['task']}, {len(X):,} rows)", flush=True)
        store.setdefault(name, {})
        for mname, (est, scale) in _model_scales(meta["task"]).items():
            if mname in store[name]:
                print(f"    {mname}: cached")
                continue
            if mname in MAX_TRAIN_ROWS and len(X) > MAX_TRAIN_ROWS[mname] * 1.25:
                print(f"    {mname}: skipped (scale cap)")
                continue
            best = tune_one(X, y, meta, mname, scale)
            if best is not None:
                store[name][mname] = best
                PARAMS_JSON.write_text(json.dumps(store, indent=2),
                                       encoding="utf-8")
    print("tuned_params.json complete")


def _model_scales(task: str) -> dict:
    lin = "elasticnet" if task == "regression" else "logreg"
    return {lin: (None, True), "knn": (None, True), "rf": (None, False),
            "mlp": (None, True), "xgboost": (None, False),
            "lightgbm": (None, False)}


def append_row(row: dict) -> None:
    exists = TUNED_CSV.exists()
    TUNED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TUNED_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_evaluation(seeds, datasets=None, protocols=None, out=None) -> None:
    from sklearn.metrics import (accuracy_score, f1_score, r2_score,
                                 mean_absolute_error)
    out_csv = Path(out) if out else TUNED_CSV
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    store = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    done = set()
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        done = set(zip(old.dataset, old.protocol, old.model, old.seed))
    for name, meta in manifest.items():
        if datasets and name not in datasets:
            continue
        if name not in store:
            continue
        df = pd.read_parquet(ROOT / meta["processed_file"])
        X, y = build_xy(df, meta)
        all_protocols = ["random"] + (["temporal"] if meta.get("temporal") else []) \
            + (["cross_site"] if meta.get("group_col") else [])
        use = [p for p in all_protocols if not protocols or p in protocols]
        print(f"== eval-tuned {name} protocols={use}", flush=True)
        for protocol in use:
            for seed in seeds:
                tr, te = make_split(df, meta, protocol, seed)
                Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
                for mname in store[name]:
                    if mname in MAX_TRAIN_ROWS and len(tr) > MAX_TRAIN_ROWS[mname]:
                        continue
                    if (name, protocol, mname, seed) in done:
                        continue
                    params = dict(store[name][mname])
                    if "hidden_layer_sizes" in params:
                        params["hidden_layer_sizes"] = tuple(
                            params["hidden_layer_sizes"])
                    est = build(mname, meta["task"], params)
                    scale = mname in ("elasticnet", "logreg", "knn", "mlp")
                    pipe = Pipeline([("prep", make_preprocessor(meta, scale)),
                                     ("model", est)])
                    pipe.fit(Xtr, ytr)
                    if meta["task"] == "regression":
                        pred = pipe.predict(Xte)
                        metrics = {"rmse": root_mean_squared_error(yte, pred),
                                   "mae": mean_absolute_error(yte, pred),
                                   "r2": r2_score(yte, pred)}
                    else:
                        proba = pipe.predict_proba(Xte)[:, 1]
                        pred = (proba >= 0.5).astype(int)
                        metrics = {"auc": roc_auc_score(yte, proba),
                                   "accuracy": accuracy_score(yte, pred),
                                   "f1": f1_score(yte, pred)}
                    row = {"dataset": name, "protocol": protocol,
                           "model": mname, "seed": seed,
                           "task": meta["task"],
                           "main_metric": "rmse" if meta["task"] == "regression"
                           else "auc",
                           "n_train": len(tr), "n_test": len(te),
                           "fit_sec": "", "pred_sec": "", "wall_sec": "",
                           **metrics}
                    exists = out_csv.exists()
                    out_csv.parent.mkdir(parents=True, exist_ok=True)
                    with out_csv.open("a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=FIELDS)
                        if not exists:
                            w.writeheader()
                        w.writerow(row)
                print(f"  [ok] {protocol}/s{seed}", flush=True)
    print(f"evaluation complete -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--protocols", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.evaluate:
        run_evaluation([int(s) for s in args.seeds.split(",")],
                       datasets=(set(args.datasets.split(","))
                                 if args.datasets else None),
                       protocols=(set(args.protocols.split(","))
                                  if args.protocols else None),
                       out=args.out)
    else:
        run_tuning()
