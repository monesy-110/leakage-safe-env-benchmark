"""Merge the parallel rain-evaluation shards into results/tuned_results.csv,
then verify the expected row count per dataset/protocol/model/seed.
Safe to re-run: dedupes on (dataset, protocol, model, seed).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAIN = ROOT / "results" / "tuned_results.csv"
SHARDS = [MAIN, RESULTS / "tuned_rain_T.csv", RESULTS / "tuned_rain_C.csv"]

frames = [pd.read_csv(p) for p in SHARDS if p.exists()]
merged = pd.concat(frames, ignore_index=True)
key = ["dataset", "protocol", "model", "seed"]
before = len(merged)
merged = merged.drop_duplicates(subset=key, keep="first").sort_values(key)
merged.to_csv(MAIN, index=False)
print(f"rows: {before} -> {len(merged)} after dedupe")

manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
expected = 0
for name, meta in manifest.items():
    protos = 1 + int(bool(meta.get("temporal"))) + int(bool(meta.get("group_col")))
    n_models = len(pd.read_csv(MAIN)[pd.read_csv(MAIN).dataset == name]
                   .model.unique())
    expected += 0  # filled below
counts = merged.groupby("dataset").size()
print(counts.to_string())
gaps = []
for name, meta in manifest.items():
    protos = 1 + int(bool(meta.get("temporal"))) + int(bool(meta.get("group_col")))
    sub = merged[merged.dataset == name]
    n_models = sub.model.nunique()
    want = n_models * protos * 10
    got = len(sub)
    if got != want:
        gaps.append(f"{name}: {got}/{want}")
print("gaps:", gaps if gaps else "none — tuned evaluation is complete")
