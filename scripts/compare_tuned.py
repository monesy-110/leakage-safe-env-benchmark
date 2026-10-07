"""Compare default vs equal-budget tuned results -> reports/tuned_comparison.md

For every (dataset, protocol, model) present in both result files, reports the
main metric (mean over seeds) under default and tuned hyperparameters, plus
the delta. Interpretation: for RMSE lower is better (negative delta improves);
for AUC higher is better (positive delta improves).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = {"regression": "rmse", "classification": "auc"}


def main() -> None:
    d = pd.read_csv(ROOT / "results" / "benchmark_results.csv")
    t = pd.read_csv(ROOT / "results" / "tuned_results.csv")
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    lines = ["# Default vs equal-budget tuned configurations", "",
             "Main metric, mean over 10 seeds. Tuned = Optuna (15 trials, 420 s "
             "cap) under a random-protocol validation split — the leaky default "
             "used in practice — then evaluated under every protocol. "
             "RMSE: negative delta improves; AUC: positive delta improves.", ""]
    for ds, meta in manifest.items():
        task = meta["task"]
        metric = MAIN[task]
        sub_d = d[d.dataset == ds]
        sub_t = t[t.dataset == ds]
        if sub_t.empty:
            continue
        g_d = sub_d.groupby(["protocol", "model"])[metric].mean()
        g_t = sub_t.groupby(["protocol", "model"])[metric].mean()
        lines += [f"## {ds} ({task}, metric = {metric})", ""]
        protos = [p for p in ("random", "temporal", "cross_site")
                  if p in g_d.index.get_level_values("protocol")
                  and p in g_t.index.get_level_values("protocol")]
        for proto in protos:
            sd, st = g_d.xs(proto, level="protocol"), g_t.xs(proto, level="protocol")
            common = sd.index.intersection(st.index)
            lines += [f"**{proto}**", "", "| model | default | tuned | delta |",
                      "|---|---|---|---|"]
            for m in common:
                lines.append(f"| {m} | {sd[m]:.4f} | {st[m]:.4f} "
                             f"| {st[m] - sd[m]:+.4f} |")
            lines.append("")
    (ROOT / "reports" / "tuned_comparison.md").write_text("\n".join(lines),
                                                          encoding="utf-8")
    print("tuned_comparison.md written")


if __name__ == "__main__":
    main()
