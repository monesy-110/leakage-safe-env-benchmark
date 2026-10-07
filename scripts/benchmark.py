"""Leakage-safe benchmark: models x split-protocols x seeds.

Reads data/manifest.json + data/processed/*.parquet, appends one row per
(dataset, protocol, model, seed) to results/benchmark_results.csv.
Already-run combinations are skipped, so the script is resumable.

Usage:
  python scripts/benchmark.py                                    # everything (long)
  python scripts/benchmark.py --datasets forest_fires --seeds 0  # smoke test
  python scripts/benchmark.py --models lightgbm --protocols cross_site

Protocols:
  random     - 80/20 holdout (stratified for classification)
  temporal   - last 20% of the time axis as test (leakage-free alternative)
  cross_site - GroupShuffleSplit on station/location (25% of groups as test)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, root_mean_squared_error, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark_results.csv"

# models that get too slow on the big datasets with default settings
MAX_TRAIN_ROWS = {"knn": 60_000, "mlp": 120_000}


def add_calendar(df: pd.DataFrame, dtc: str) -> pd.DataFrame:
    dt = df[dtc]
    out = pd.DataFrame(index=df.index)
    m = dt.dt.month
    out["cal_month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    out["cal_month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)
    out["cal_dow"] = dt.dt.dayofweek.astype("float64")
    h = dt.dt.hour
    if h.nunique() > 1:
        out["cal_hour_sin"] = np.sin(2 * np.pi * h / 24)
        out["cal_hour_cos"] = np.cos(2 * np.pi * h / 24)
    return out


def build_xy(df: pd.DataFrame, meta: dict):
    X = df[meta["features"]].copy()
    cat = meta["categorical_features"]
    X[cat] = X[cat].fillna("missing").astype(str)
    if meta.get("datetime_col") and meta.get("temporal"):
        X = pd.concat([X, add_calendar(df, meta["datetime_col"])], axis=1)
    return X, df[meta["target"]]


def make_split(df: pd.DataFrame, meta: dict, protocol: str, seed: int):
    idx = np.arange(len(df))
    if protocol == "random":
        strat = df[meta["target"]] if meta["task"] == "classification" else None
        return train_test_split(idx, test_size=0.2, random_state=seed,
                                stratify=strat)
    if protocol == "temporal":
        dt = df[meta["datetime_col"]]
        cutoff = dt.quantile(0.8)
        return idx[dt < cutoff], idx[dt >= cutoff]
    if protocol == "cross_site":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        return next(gss.split(df, groups=df[meta["group_col"]]))
    raise ValueError(protocol)


def make_preprocessor(meta: dict, scale: bool):
    num = [c for c in meta["features"] if c not in meta["categorical_features"]]
    cat = meta["categorical_features"]
    num_pipe = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_pipe.append(("scale", StandardScaler()))
    ct = ColumnTransformer([
        ("num", Pipeline(num_pipe), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    return ct


def make_models(task: str, seed: int):
    """name -> (estimator, scale?)"""
    import lightgbm as lgb
    import xgboost as xgb

    if task == "regression":
        lin = ElasticNet(alpha=0.1, max_iter=20000)
        knn = KNeighborsRegressor(n_neighbors=15)
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1)
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300,
                           early_stopping=True, random_state=seed)
        xgbm = xgb.XGBRegressor(n_estimators=400, learning_rate=0.06,
                                tree_method="hist", random_state=seed, n_jobs=-1)
        lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.06,
                                 random_state=seed, n_jobs=-1, verbosity=-1)
    else:
        lin = LogisticRegression(max_iter=5000)
        knn = KNeighborsClassifier(n_neighbors=15)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                    random_state=seed, n_jobs=-1)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                            early_stopping=True, random_state=seed)
        xgbm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.06,
                                 tree_method="hist", eval_metric="logloss",
                                 random_state=seed, n_jobs=-1)
        lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.06,
                                  random_state=seed, n_jobs=-1, verbosity=-1)
    return {"elasticnet" if task == "regression" else "logreg": (lin, True),
            "knn": (knn, True), "rf": (rf, False), "mlp": (mlp, True),
            "xgboost": (xgbm, False), "lightgbm": (lgbm, False)}


def evaluate(task: str, model, Xtr, ytr, Xte, yte):
    t0 = time.time()
    model.fit(Xtr, ytr)
    fit_sec = time.time() - t0
    t0 = time.time()
    if task == "regression":
        pred = model.predict(Xte)
        metrics = {"rmse": root_mean_squared_error(yte, pred),
                   "mae": mean_absolute_error(yte, pred),
                   "r2": r2_score(yte, pred)}
    else:
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {"auc": roc_auc_score(yte, proba),
                   "accuracy": accuracy_score(yte, pred),
                   "f1": f1_score(yte, pred)}
    return metrics, fit_sec, time.time() - t0


# fixed schema: classification rows leave rmse/mae/r2 empty and vice versa;
# a fixed list prevents silent column misalignment when tasks are interleaved
FIELDS = ["dataset", "protocol", "model", "seed", "task", "main_metric",
          "n_train", "n_test", "fit_sec", "pred_sec", "wall_sec",
          "rmse", "mae", "r2", "auc", "accuracy", "f1"]


def append_row(row: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def done_keys() -> set:
    if not RESULTS.exists():
        return set()
    df = pd.read_csv(RESULTS)
    return set(zip(df.dataset, df.protocol, df.model, df.seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--models", default="all")
    ap.add_argument("--protocols", default="all")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    want_d = None if args.datasets == "all" else set(args.datasets.split(","))
    want_m = None if args.models == "all" else set(args.models.split(","))
    want_p = None if args.protocols == "all" else set(args.protocols.split(","))

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    done = done_keys()

    for name, meta in manifest.items():
        if want_d and name not in want_d:
            continue
        protocols = ["random"]
        if meta.get("temporal"):
            protocols.append("temporal")
        if meta.get("group_col"):
            protocols.append("cross_site")
        df = pd.read_parquet(ROOT / meta["processed_file"])
        X, y = build_xy(df, meta)
        print(f"\n== {name} ({meta['task']}, {len(df):,} rows) protocols={protocols}",
              flush=True)
        for protocol in protocols:
            if want_p and protocol not in want_p:
                continue
            for seed in seeds:
                tr, te = make_split(df, meta, protocol, seed)
                Xtr, Xte, ytr, yte = (X.iloc[tr], X.iloc[te],
                                      y.iloc[tr], y.iloc[te])
                for mname, (est, scale) in make_models(meta["task"], seed).items():
                    if want_m and mname not in want_m:
                        continue
                    if (name, protocol, mname, seed) in done:
                        print(f"  [skip done] {protocol}/{mname}/s{seed}")
                        continue
                    if mname in MAX_TRAIN_ROWS and len(tr) > MAX_TRAIN_ROWS[mname]:
                        print(f"  [skip big ] {protocol}/{mname}/s{seed} "
                              f"(train {len(tr):,} > {MAX_TRAIN_ROWS[mname]:,})")
                        continue
                    pipe = Pipeline([("prep", make_preprocessor(meta, scale)),
                                     ("model", est)])
                    t0 = time.time()
                    try:
                        metrics, fit_sec, pred_sec = evaluate(
                            meta["task"], pipe, Xtr, ytr, Xte, yte)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [FAIL] {protocol}/{mname}/s{seed}: {e!r}")
                        continue
                    row = {"dataset": name, "protocol": protocol,
                           "model": mname, "seed": seed, "task": meta["task"],
                           "main_metric": "rmse" if meta["task"] == "regression"
                           else "auc",
                           "n_train": len(tr), "n_test": len(te),
                           "fit_sec": round(fit_sec, 2),
                           "pred_sec": round(pred_sec, 2),
                           "wall_sec": round(time.time() - t0, 2), **metrics}
                    append_row(row)
                    print(f"  [ok] {protocol}/{mname}/s{seed} "
                          f"({', '.join(f'{k}={v:.4f}' for k, v in metrics.items())})",
                          flush=True)


if __name__ == "__main__":
    main()
