"""FT-Transformer baseline (Gorishniy et al., 2021: Feature Tokenizer +
Transformer encoder with a [CLS] token), added to answer the reviewer
question ``which deep model represents deep learning?''.

Self-contained relative to benchmark.py: reuses build_xy / make_split /
FIELDS. Categorical features are ordinal-encoded against a train-fitted
vocabulary (+1 reserved for unseen values); numeric features are
median-imputed and standardised on the training fold only. The model is a
small CPU-sized configuration: d_model = 32, 2 encoder layers, 4 heads.

Results are appended to results/ft_results.csv using the same fixed-field
schema as benchmark_results.csv, with model = "fttransformer". Resumable via
the usual (dataset, protocol, model, seed) key check.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, root_mean_squared_error, roc_auc_score)

from benchmark import ROOT, build_xy, make_split, FIELDS  # noqa: E402

RESULTS_CSV = ROOT / "results" / "ft_results.csv"
DEVICE = torch.device("cpu")


class FeatureTokenizer(nn.Module):
    def __init__(self, n_num, cat_cards, d):
        super().__init__()
        self.n_num = n_num
        self.num_weight = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(n_num, d))
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(c, d) for c in cat_cards])
        self.cat_bias = nn.Parameter(torch.zeros(len(cat_cards), d))

    def forward(self, xnum, xcat):
        tokens = xnum.unsqueeze(-1) * self.num_weight + self.num_bias
        if self.cat_embeddings:
            emb = torch.stack([emb_i(xcat[:, i])
                               for i, emb_i in enumerate(self.cat_embeddings)],
                              dim=1)
            tokens = torch.cat([tokens, emb + self.cat_bias], dim=1)
        return tokens


class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cards, d=32, n_layers=2, n_heads=4,
                 ffn=128, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cards, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=ffn, dropout=dropout,
            batch_first=True, activation="relu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d, 1)

    def forward(self, xnum, xcat):
        tokens = self.tokenizer(xnum, xcat)
        cls = self.cls.expand(len(tokens), -1, -1)
        z = self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]
        return self.head(z).squeeze(-1)


def encode_features(Xtr, Xte, cat_features):
    """Median-impute + standardise numerics; ordinal-encode categoricals with
    a train-fitted vocabulary (0 reserved for unseen values)."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    num = [c for c in Xtr.columns if c not in cat_features]
    med = Xtr[num].median()
    Xtr[num] = Xtr[num].fillna(med)
    Xte[num] = Xte[num].fillna(med)
    mu, sd = Xtr[num].mean(), Xtr[num].std().replace(0, 1)
    Xtr[num] = (Xtr[num] - mu) / sd
    Xte[num] = (Xte[num] - mu) / sd
    cards, tr_cat, te_cat = [], [], []
    for c in cat_features:
        levels = {v: i + 1 for i, v in enumerate(sorted(Xtr[c].astype(str)
                                                        .unique()))}
        cards.append(len(levels) + 1)
        tr_cat.append(Xtr[c].astype(str).map(levels).fillna(0).to_numpy())
        te_cat.append(Xte[c].astype(str).map(levels).fillna(0).to_numpy())
    Xtr_t = (torch.tensor(Xtr[num].to_numpy(np.float32)),
             torch.tensor(np.stack(tr_cat, axis=1), dtype=torch.long)
             if tr_cat else None)
    Xte_t = (torch.tensor(Xte[num].to_numpy(np.float32)),
             torch.tensor(np.stack(te_cat, axis=1), dtype=torch.long)
             if te_cat else None)
    return Xtr_t, Xte_t, cards


def fit_ft(Xtr_t, ytr, task, cards, seed, max_epochs=25, patience=3,
           batch=1024, lr=1e-3):
    torch.manual_seed(seed)
    model = FTTransformer(Xtr_t[0].shape[1], cards).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    n = len(Xtr_t[0])
    idx = torch.randperm(n)
    n_val = max(1, int(0.1 * n))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for lo in range(0, len(perm), batch):
            b = perm[lo:lo + batch]
            xnum, xcat = Xtr_t[0][b], (Xtr_t[1][b] if Xtr_t[1] is not None
                                       else None)
            out = model(xnum, xcat)
            loss = lossf(out, ytr[b] if task == "regression" else ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xtr_t[0][va_idx],
                        Xtr_t[1][va_idx] if Xtr_t[1] is not None else None)
            vloss = lossf(out, ytr[va_idx]).item()
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def append_row(row, out_csv=None):
    out_csv = Path(out_csv) if out_csv else RESULTS_CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def done_keys(out_csv=None):
    out_csv = Path(out_csv) if out_csv else RESULTS_CSV
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    return set(zip(df.dataset, df.protocol, df.model, df.seed))


_CARDINALITIES = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--protocols", default="all")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out_csv = args.out
    manifest = json.loads((ROOT / "data" / "manifest.json")
                          .read_text(encoding="utf-8"))
    done = done_keys(out_csv)
    for name in args.datasets.split(","):
        meta = manifest[name]
        df = pd.read_parquet(ROOT / meta["processed_file"])
        X, y = build_xy(df, meta)
        cat_features = meta["categorical_features"]
        all_protocols = ["random"] + \
            (["temporal"] if meta.get("temporal") else []) + \
            (["cross_site"] if meta.get("group_col") else [])
        protocols = (all_protocols if args.protocols == "all"
                     else [p for p in args.protocols.split(",")
                           if p in all_protocols])
        for protocol in protocols:
            for seed in seeds:
                if (name, protocol, "fttransformer", seed) in done:
                    print(f"  [skip done] {protocol}/s{seed}", flush=True)
                    continue
                tr, te = make_split(df, meta, protocol, seed)
                Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], \
                    y.iloc[te]
                Xtr_t, Xte_t, cards = encode_features(
                    Xtr, Xte, cat_features)
                ytr_t = torch.tensor(
                    ytr.to_numpy(np.float32))
                t0 = time.time()
                model = fit_ft(Xtr_t, ytr_t, meta["task"], cards, seed)
                fit_sec = time.time() - t0
                model.eval()
                with torch.no_grad():
                    out = model(Xte_t[0], Xte_t[1]).numpy()
                if meta["task"] == "regression":
                    pred = out
                    metrics = {"rmse": root_mean_squared_error(yte, pred),
                               "mae": mean_absolute_error(yte, pred),
                               "r2": r2_score(yte, pred)}
                else:
                    proba = 1 / (1 + np.exp(-out))
                    pred = (proba >= 0.5).astype(int)
                    metrics = {"auc": roc_auc_score(yte, proba),
                               "accuracy": accuracy_score(yte, pred),
                               "f1": f1_score(yte, pred)}
                row = {"dataset": name, "protocol": protocol,
                       "model": "fttransformer", "seed": seed,
                       "task": meta["task"],
                       "main_metric": "rmse" if meta["task"] == "regression"
                       else "auc",
                       "n_train": len(tr), "n_test": len(te),
                       "fit_sec": round(fit_sec, 2), "pred_sec": "",
                       "wall_sec": round(fit_sec, 2), **metrics}
                append_row(row, out_csv)
                print(f"  [ok] {protocol}/s{seed} "
                      f"({', '.join(f'{k}={v:.4f}' for k, v in metrics.items())})",
                      flush=True)


if __name__ == "__main__":
    main()
