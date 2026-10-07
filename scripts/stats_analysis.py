"""Statistical analysis of benchmark results -> reports/stats_report.md

Per (dataset, protocol):
  - mean +/- std of the main metric per model
  - Friedman test across models (blocks = seeds) + Nemenyi critical difference

Global:
  - leakage inflation: metric(random) - metric(temporal) per dataset x model,
    using higher-is-better metrics (R2 for regression, AUC for classification)
  - transfer gap: metric(random) - metric(cross_site)
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark_results.csv"
REPORTS = ROOT / "reports"

Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
       7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
MAIN = {"regression": ("rmse", True), "classification": ("auc", False)}
HIGHER = {"regression": "r2", "classification": "auc"}


def pivot(df: pd.DataFrame, ds: str, proto: str, metric: str) -> pd.DataFrame:
    sub = df[(df.dataset == ds) & (df.protocol == proto)]
    if sub.empty:
        return pd.DataFrame()
    p = sub.pivot_table(index="seed", columns="model", values=metric)
    return p.dropna(axis=0, how="any").dropna(axis=1, how="any")


def friedman_nemenyi(piv: pd.DataFrame, lower_better: bool):
    models = list(piv.columns)
    if len(models) < 3 or len(piv) < 3:
        return None
    if piv.nunique().le(1).all():
        return None  # identical values across blocks (deterministic split)
    chi2, p = stats.friedmanchisquare(*[piv[m].values for m in models])
    ranks = piv.rank(axis=1, ascending=lower_better)
    avg = ranks.mean()
    k, n = len(models), len(piv)
    cd = Q05[k] * math.sqrt(k * (k + 1) / (6 * n)) if k in Q05 else float("nan")
    sig = [(a, b, float(abs(avg[a] - avg[b])))
           for i, a in enumerate(models) for b in models[i + 1:]
           if abs(avg[a] - avg[b]) > cd]
    return {"chi2": chi2, "p": p, "avg_rank": avg.to_dict(), "cd": cd,
            "n_sig_pairs": len(sig)}


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float, np.floating)) else str(v)


def main() -> None:
    df = pd.read_csv(RESULTS)
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    order = list(manifest)  # keep manifest order
    out = ["# Benchmark statistical report", "",
           "Main metric: RMSE (regression, lower=better) / AUC (classification). "
           "Seeds are the repeated-measurement blocks for Friedman.", ""]

    # ---- per dataset x protocol ----
    out += ["## Per-dataset results", ""]
    for ds in order:
        task = manifest[ds]["task"]
        metric, lower = MAIN[task]
        out += [f"### {ds} ({task})", ""]
        protos = [p for p in df[df.dataset == ds].protocol.unique()]
        for proto in protos:
            piv = pivot(df, ds, proto, metric)
            if piv.empty:
                continue
            out.append(f"**{proto}** (n_seeds={len(piv)})")
            out += ["", "| model | mean | std | avg rank |", "|---|---|---|---|"]
            fn = friedman_nemenyi(piv, lower)
            ranks = fn["avg_rank"] if fn else {}
            for m in piv.columns:
                out.append(f"| {m} | {piv[m].mean():.4f} | {piv[m].std():.4f} "
                           f"| {ranks.get(m, float('nan')):.2f} |")
            if fn:
                out += ["", f"Friedman chi2={fn['chi2']:.2f}, p={fn['p']:.2e}; "
                            f"Nemenyi CD={fn['cd']:.3f}; "
                            f"significantly different pairs: {fn['n_sig_pairs']}"]
            out.append("")
        out.append("")

    # ---- leakage inflation (random vs temporal) ----
    out += ["## Leakage inflation: metric(random) - metric(temporal)", "",
            "Higher-is-better metrics (R2 for regression, AUC for classification); "
            "positive = random split overestimates generalization.", ""]
    models = sorted(df.model.unique())
    inf_rows = []
    for ds in order:
        task = manifest[ds]["task"]
        m = HIGHER[task]
        r = pivot(df, ds, "random", m).mean()
        t = pivot(df, ds, "temporal", m).mean()
        if r.empty or t.empty:
            continue
        common = r.index.intersection(t.index)
        inf_rows.append({"dataset": ds,
                         **{mname: r[mname] - t[mname] for mname in common}})
        out.append(f"- **{ds}**: " + ", ".join(
            f"{mname} {r[mname] - t[mname]:+.3f}" for mname in common))
    inf = pd.DataFrame(inf_rows).set_index("dataset") if inf_rows else pd.DataFrame()
    out.append("")
    if not inf.empty:
        by_model = inf.mean(numeric_only=True).sort_values(ascending=False)
        out += ["Mean inflation by model (across datasets): " + ", ".join(
            f"{m} {v:+.3f}" for m, v in by_model.items()), ""]

    # ---- transfer gap (random vs cross_site) ----
    out += ["## Transfer gap: metric(random) - metric(cross_site)", ""]
    for ds in order:
        task = manifest[ds]["task"]
        m = HIGHER[task]
        r = pivot(df, ds, "random", m).mean()
        c = pivot(df, ds, "cross_site", m).mean()
        if r.empty or c.empty:
            continue
        common = r.index.intersection(c.index)
        out.append(f"- **{ds}**: " + ", ".join(
            f"{mname} {r[mname] - c[mname]:+.3f}" for mname in common))
    out.append("")

    (REPORTS / "stats_report.md").write_text("\n".join(out), encoding="utf-8")
    if not inf.empty:
        inf.to_csv(REPORTS / "leakage_inflation.csv")
    print("stats_report.md written;",
          f"{len(inf)} datasets with leakage-inflation analysis")


if __name__ == "__main__":
    main()
