"""Cross-check every key number quoted in paper_draft.md against the CSVs.

Each check prints PASS/FAIL with the recomputed value. Run after any data
refresh; all claims in the paper must pass before submission.
"""
import pandas as pd

R = "results/"
b = pd.read_csv(R + "benchmark_results.csv")
t = pd.read_csv(R + "tuned_results.csv")
results = []


def check(label, actual, expect, tol=0.0005):
    ok = abs(actual - expect) <= tol
    results.append((label, ok, actual, expect))


def r2_inflation(df, ds, model):
    s = df[df.dataset == ds]
    return (s[s.protocol == "random"].groupby("model").r2.mean()[model]
            - s[s.protocol == "temporal"].groupby("model").r2.mean()[model])


# --- Table 2: beijing_pm25 R2 random/temporal/inflation (default) ---
pm25_expect = {  # model: (random, temporal, inflation)
    "mlp": (0.681, -0.140, 0.821), "rf": (0.842, 0.483, 0.359),
    "lightgbm": (0.775, 0.494, 0.281), "xgboost": (0.759, 0.497, 0.262),
    "knn": (0.528, 0.288, 0.240), "elasticnet": (0.263, 0.233, 0.029)}
for m, (rd, tp, inf) in pm25_expect.items():
    s = b[(b.dataset == "beijing_pm25")]
    rd_a = s[s.protocol == "random"].groupby("model").r2.mean()[m]
    tp_a = s[s.protocol == "temporal"].groupby("model").r2.mean()[m]
    check(f"pm25 {m} random", rd_a, rd, 0.001)
    check(f"pm25 {m} temporal", tp_a, tp, 0.001)
    check(f"pm25 {m} inflation", rd_a - tp_a, inf, 0.001)

# --- tuned pm25 inflation (MLP 2.800, kNN 0.584, XGB 0.481, LGBM 0.456) ---
for m, expect in [("mlp", 2.800), ("knn", 0.584), ("xgboost", 0.481),
                  ("lightgbm", 0.456), ("rf", 0.377)]:
    check(f"pm25 tuned inflation {m}", r2_inflation(t, "beijing_pm25", m),
          expect, 0.001)
check("pm25 tuned MLP temporal R2",
      t[(t.dataset == "beijing_pm25") & (t.protocol == "temporal")
        & (t.model == "mlp")].r2.mean(), -2.010, 0.001)

# --- multisite RMSE: RF 17.0/19.1/21.0; enet 31.8->25.1; xgb temporal 24.80->20.11 ---
ms = b[b.dataset == "beijing_multisite"]
check("multisite RF random RMSE", ms[(ms.protocol == "random") & (ms.model == "rf")].rmse.mean(), 17.00, 0.005)
check("multisite RF cross RMSE", ms[(ms.protocol == "cross_site") & (ms.model == "rf")].rmse.mean(), 19.12, 0.005)
check("multisite RF temporal RMSE", ms[(ms.protocol == "temporal") & (ms.model == "rf")].rmse.mean(), 20.98, 0.005)
check("multisite ENet temporal RMSE", ms[(ms.protocol == "temporal") & (ms.model == "elasticnet")].rmse.mean(), 25.13, 0.005)
tm = t[t.dataset == "beijing_multisite"]
check("multisite tuned XGB temporal RMSE", tm[(tm.protocol == "temporal") & (tm.model == "xgboost")].rmse.mean(), 20.11, 0.005)

# --- rain AUC: MLP 0.897/0.886, XGB cross 0.876, logreg range, inflation, tuned inflation ---
rain = b[b.dataset == "rain_australia"]
check("rain MLP random AUC", rain[(rain.protocol == "random") & (rain.model == "mlp")].auc.mean(), 0.8972, 0.0005)
check("rain MLP temporal AUC", rain[(rain.protocol == "temporal") & (rain.model == "mlp")].auc.mean(), 0.8865, 0.0005)
check("rain XGB cross AUC", rain[(rain.protocol == "cross_site") & (rain.model == "xgboost")].auc.mean(), 0.8760, 0.0005)
inf_rain = {m: (rain[(rain.protocol == "random") & (rain.model == m)].auc.mean()
                - rain[(rain.protocol == "temporal") & (rain.model == m)].auc.mean())
            for m in rain.model.unique()}
check("rain inflation range low", min(inf_rain.values()), 0.0100, 0.0005)
check("rain inflation range high", max(inf_rain.values()), 0.0130, 0.0005)
tr = t[t.dataset == "rain_australia"]
inf_rain_t = {m: (tr[(tr.protocol == "random") & (tr.model == m)].auc.mean()
                  - tr[(tr.protocol == "temporal") & (tr.model == m)].auc.mean())
              for m in tr.model.unique()}
check("rain tuned inflation low", min(inf_rain_t.values()), 0.0113, 0.0005)
check("rain tuned inflation high", max(inf_rain_t.values()), 0.0171, 0.0005)

# --- saturation claims: tree/MLP R2 >= 0.99 on air/steel; knn+linear lower ---
for ds in ["air_quality", "steel_energy"]:
    s = b[b.dataset == ds]
    trees = s[s.model.isin(["lightgbm", "mlp", "rf", "xgboost"])].groupby(
        ["protocol", "model"]).r2.mean()
    low = s[s.model.isin(["knn", "elasticnet"])].groupby(
        ["protocol", "model"]).r2.mean()
    check(f"{ds} tree/MLP min R2 >= 0.99", min(trees.min(), 0.99), 0.99, 0.0)
    check(f"{ds} knn/linear min R2 in [0.90,0.98]", low.min(), 0.94, 0.04)

# --- FT-Transformer (modern tabular deep baseline) ---
ft = pd.read_csv(R + "ft_results.csv")
check("FT rows", len(ft), 51, 0)
fpm = ft[ft.dataset == "beijing_pm25"]
check("pm25 FT random R2", fpm[fpm.protocol == "random"].r2.mean(), 0.5932, 0.001)
check("pm25 FT temporal R2", fpm[fpm.protocol == "temporal"].r2.mean(), 0.4160, 0.001)
check("pm25 FT inflation", fpm[fpm.protocol == "random"].r2.mean()
      - fpm[fpm.protocol == "temporal"].r2.mean(), 0.1772, 0.001)
fr = ft[ft.dataset == "rain_australia"]
check("rain FT random AUC", fr[fr.protocol == "random"].auc.mean(), 0.8932, 0.001)
check("rain FT temporal AUC", fr[fr.protocol == "temporal"].auc.mean(), 0.8827, 0.001)
check("rain FT cross AUC", fr[fr.protocol == "cross_site"].auc.mean(), 0.8706, 0.001)
check("rain FT inflation", fr[fr.protocol == "random"].auc.mean()
      - fr[fr.protocol == "temporal"].auc.mean(), 0.0105, 0.001)

# --- counts ---
check("default rows 810", len(b), 810, 0)
check("tuned rows 810", len(t), 810, 0)
check("seeds 10", b.seed.nunique(), 10, 0)

# --- report ---
fails = [r for r in results if not r[1]]
for label, ok, actual, expect in results:
    print(f"{'PASS' if ok else 'FAIL':4s} {label:36s} actual={actual:+.4f} expect={expect:+.4f}")
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
if fails:
    raise SystemExit(1)
