"""Journal-grade figures (Nature-style) -> reports/figures/*.png, 300 dpi.

Design rules:
  - Arial, font sizes set for the FINAL printed size (no shrink-to-fit)
  - no top/right spines, faint gridlines, muted colour-blind-safe palette
  - panel letters (a)-(h) on the multi-panel figure
  - Cleveland dot-plus-error style instead of heavy grouped bars
  - FT-Transformer runs (results/ft_results.csv) merged into every figure
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# figures are English-only; override the user-level CJK matplotlibrc
# (SimHei etc. lack the U+2212 minus glyph)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.linewidth"] = 0.6
matplotlib.rcParams["xtick.major.width"] = 0.6
matplotlib.rcParams["ytick.major.width"] = 0.6
matplotlib.rcParams["xtick.major.size"] = 2.5
matplotlib.rcParams["ytick.major.size"] = 2.5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark_results.csv"
FIGDIR = ROOT / "reports" / "figures"

MAIN = {"regression": ("rmse", "RMSE (lower is better)"),
        "classification": ("auc", "ROC-AUC")}
COLOR = {"random": "#4477AA", "temporal": "#EE6677",
         "cross_site": "#228833"}  # Okabe-Ito, colour-blind safe
MARK = {"random": "o", "temporal": "^", "cross_site": "s"}
PROT_LABEL = {"random": "Random", "temporal": "Temporal",
              "cross_site": "Cross-site"}
DISPLAY = {"fttransformer": "FT-Transformer", "elasticnet": "Elastic net",
           "logreg": "Logistic reg.", "knn": "$k$-NN",
           "rf": "Random forest", "xgboost": "XGBoost",
           "lightgbm": "LightGBM", "mlp": "MLP"}
MODEL_ORDER = ["Elastic net", "Logistic reg.", "$k$-NN", "MLP",
               "FT-Transformer", "LightGBM", "XGBoost", "Random forest"]


def style_axis(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, length=2.5, width=0.6)


def load():
    df = pd.read_csv(RESULTS)
    ft_path = ROOT / "results" / "ft_results.csv"
    if ft_path.exists():
        df = pd.concat([df, pd.read_csv(ft_path)], ignore_index=True)
    df["model"] = df["model"].map(DISPLAY).fillna(df["model"])
    return df


def main() -> None:
    df = load()
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    datasets = list(manifest)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
        "legend.fontsize": 6.2, "figure.dpi": 300,
    })

    # ---- fig 1: dot-plus-error plot per dataset, panels (a)-(h) ----
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 7.6))
    letters = "abcdefgh"
    for k, (ax, ds) in enumerate(zip(axes.ravel(), datasets)):
        task = manifest[ds]["task"]
        metric, ylabel = MAIN[task]
        sub_all = df[df.dataset == ds]
        protos = [p for p in ("random", "temporal", "cross_site")
                  if p in set(sub_all.protocol)]
        models = [m for m in MODEL_ORDER if m in set(sub_all.model)]
        dodge = 0.22
        for i, proto in enumerate(protos):
            g = (sub_all[sub_all.protocol == proto]
                 .groupby("model")[metric].agg(["mean", "std"])
                 .reindex(models))
            xs = np.arange(len(models)) + (i - (len(protos) - 1) / 2) * dodge
            ax.errorbar(xs, g["mean"], yerr=g["std"], fmt=MARK[proto],
                        color=COLOR[proto], markersize=3.4,
                        elinewidth=0.7, capsize=1.6, capthick=0.7,
                        label=PROT_LABEL[proto], mec="white", mew=0.4)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=40, ha="right")
        style_axis(ax)
        ax.set_title(ds, fontsize=7.2, pad=3)
        ax.set_ylabel(ylabel, fontsize=6.8)
        ax.text(-0.14, 1.06, f"({letters[k]})", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")
        if k == 0:
            ax.legend(frameon=False, handletextpad=0.1, borderaxespad=0.1,
                      loc="best")
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.115,
                        hspace=0.62, wspace=0.24)
    fig.savefig(FIGDIR / "fig1_performance.png", dpi=300)
    fig.savefig(FIGDIR / "fig1_performance.pdf")
    plt.close(fig)

    # ---- fig 2: leakage heatmap ----
    rows = {}
    for ds in datasets:
        m = "r2" if manifest[ds]["task"] == "regression" else "auc"
        r = df[(df.dataset == ds) & (df.protocol == "random")].groupby("model")[m].mean()
        t = df[(df.dataset == ds) & (df.protocol == "temporal")].groupby("model")[m].mean()
        common = r.index.intersection(t.index)
        if len(common):
            rows[ds] = (common, r, t)
    models = [m for m in MODEL_ORDER
              if m in set().union(*[c for c, _, _ in rows.values()])] if rows else []
    if rows:
        M = np.full((len(rows), len(models)), np.nan)
        for i, (_, (common, r, t)) in enumerate(rows.items()):
            for j, mname in enumerate(models):
                if mname in common:
                    M[i, j] = r[mname] - t[mname]
        fig, ax = plt.subplots(figsize=(4.6, 0.42 * len(rows) + 1.15))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-0.85, vmax=0.85)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=38, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(list(rows), fontsize=7)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    v = M[i, j]
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6.3,
                            color="white" if abs(v) > 0.55 else "#222222")
                else:
                    ax.text(j, i, "–", ha="center", va="center",
                            fontsize=6, color="#bbbbbb")
        ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("$R^2$/AUC(random) $-$ $R^2$/AUC(temporal)", fontsize=6.8)
        cb.ax.tick_params(labelsize=6.3, width=0.6)
        cb.outline.set_linewidth(0.5)
        fig.tight_layout(pad=0.4)
        fig.savefig(FIGDIR / "fig2_leakage_heatmap.png", dpi=300)
        fig.savefig(FIGDIR / "fig2_leakage_heatmap.pdf")
        plt.close(fig)

    # ---- fig 3: computational cost ----
    med = df.groupby(["dataset", "protocol", "model"]).wall_sec.transform("median")
    cost = df[df.wall_sec <= med * 5]
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    palette = dict(zip(MODEL_ORDER,
                       ["#4477AA", "#66CCEE", "#228833", "#CCBB44",
                        "#AA3377", "#EE6677", "#AA4499", "#888888"]))
    for mname, sub in cost.groupby("model"):
        ax.scatter(sub.n_train, sub.wall_sec.clip(lower=0.05), s=11,
                   alpha=0.65, linewidths=0, label=mname,
                   color=palette.get(mname, "#888888"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axis(ax, grid_axis="both")
    ax.set_xlabel("Training rows", fontsize=7)
    ax.set_ylabel("Wall-clock seconds (fit + predict)", fontsize=7)
    ax.legend(frameon=False, fontsize=6.2, handletextpad=0.05,
              borderaxespad=0.1, loc="upper left", ncol=1)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGDIR / "fig3_cost.png", dpi=300)
    fig.savefig(FIGDIR / "fig3_cost.pdf")
    plt.close(fig)

    print("journal-grade figures written to", FIGDIR)


if __name__ == "__main__":
    main()
