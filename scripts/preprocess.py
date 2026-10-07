"""Clean + standardize the 8 raw datasets.

Outputs:
  data/processed/<name>.parquet  - typed columns, parsed datetime, NaNs preserved
  data/manifest.json             - task/target/features/split definitions per dataset
  reports/data_report.md         - data health report (rows, missingness, target stats)

Design notes:
  - Missing values are NOT imputed here; imputation is part of the experiment
    pipeline (LightGBM/XGBoost handle NaN natively, others impute in-pipeline).
  - Rows whose target is missing are dropped (count reported).
  - Known leakage columns are excluded and documented in the manifest.
"""
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SOURCES = {
    "forest_fires": "https://archive.ics.uci.edu/dataset/162/forest+fires",
    "beijing_multisite": "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
    "air_quality": "https://archive.ics.uci.edu/dataset/360/air+quality",
    "ccpp": "https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
    "beijing_pm25": "https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data",
    "steel_energy": "https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption",
    "rain_australia": "https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package",
    "water_potability": "https://www.kaggle.com/datasets/adityakadiwal/water-potability",
}


def clean_col(c: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).strip("_").lower()


def load_beijing_multisite() -> pd.DataFrame:
    d = RAW / "beijing_multisite"
    inner = next(d.glob("PRSA2017_Data*.zip"))
    folder = d / inner.stem
    if not folder.exists():
        with zipfile.ZipFile(inner) as z:
            z.extractall(d)
    files = sorted(folder.rglob("PRSA_Data_*.csv"))
    if not files:  # some UCI zips nest one level deeper
        files = sorted(d.rglob("PRSA_Data_*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df.columns = [clean_col(c) for c in df.columns]  # PM2.5 -> pm2_5
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df.drop(columns=["no"])


def load_beijing_pm25() -> pd.DataFrame:
    df = pd.read_csv(RAW / "beijing_pm25" / "PRSA_data_2010.1.1-2014.12.31.csv",
                     na_values=["NA"])
    df.columns = [clean_col(c) for c in df.columns]
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df.drop(columns=["no"])


def load_air_quality() -> pd.DataFrame:
    df = pd.read_excel(RAW / "air_quality" / "AirQualityUCI.xlsx")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [clean_col(c) for c in df.columns]  # CO(GT) -> co_gt, T -> t
    df = df.replace(-200, np.nan)  # documented sensor error code
    # date may come back as datetime64 (xlsx) or string; time is "18.00.00"
    d = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    t = pd.to_timedelta(
        df["time"].astype(str).str.replace(".", ":", regex=False), errors="coerce")
    df["datetime"] = d + t
    return df.drop(columns=["date", "time"])


def load_ccpp() -> pd.DataFrame:
    df = pd.read_excel(RAW / "ccpp" / "CCPP" / "Folds5x2_pp.xlsx")
    df.columns = [clean_col(c) for c in df.columns]
    return df


def load_steel_energy() -> pd.DataFrame:
    df = pd.read_csv(RAW / "steel_energy" / "Steel_industry_data.csv",
                     encoding="utf-8-sig")
    df.columns = [clean_col(c) for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M",
                                    errors="coerce")
    return df.drop(columns=["date"])


def load_forest_fires() -> pd.DataFrame:
    return pd.read_csv(RAW / "forest_fires" / "forestfires.csv")


def load_water_potability() -> pd.DataFrame:
    df = pd.read_csv(RAW / "water_potability" / "water_potability.csv")
    df.columns = [clean_col(c) for c in df.columns]
    return df


def load_rain_australia() -> pd.DataFrame:
    df = pd.read_csv(RAW / "rain_australia" / "weatherAUS.csv", na_values=["NA"])
    df.columns = [clean_col(c) for c in df.columns]
    for c in ("raintoday", "raintomorrow"):
        df[c] = df[c].map({"No": 0, "Yes": 1})
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    return df.drop(columns=["date"])


LOADERS = {
    "beijing_multisite": load_beijing_multisite,
    "beijing_pm25": load_beijing_pm25,
    "air_quality": load_air_quality,
    "ccpp": load_ccpp,
    "steel_energy": load_steel_energy,
    "forest_fires": load_forest_fires,
    "water_potability": load_water_potability,
    "rain_australia": load_rain_australia,
}

TASKS = {
    "beijing_multisite": dict(task="regression", target="pm2_5",
                              datetime_col="datetime",
                              group_col="station", temporal=True, cross_site=True,
                              excluded={}, notes="12 stations, 2013-2017 hourly"),
    "beijing_pm25": dict(task="regression", target="pm2_5",
                         datetime_col="datetime",
                         group_col=None, temporal=True, cross_site=False,
                         excluded={}, notes="single site, 2010-2014 hourly"),
    "air_quality": dict(task="regression", target="c6h6_gt",
                        datetime_col="datetime",
                        group_col=None, temporal=True, cross_site=False,
                        excluded={}, notes="-200 sensor error code set to NaN; "
                        "alternative targets: co_gt, no2_gt, t"),
    "ccpp": dict(task="regression", target="pe", group_col=None,
                 temporal=False, cross_site=False,
                 excluded={}, notes="no time index; 4 ambient features -> net output"),
    "steel_energy": dict(task="regression", target="usage_kwh",
                         datetime_col="datetime",
                         group_col=None, temporal=True, cross_site=False,
                         excluded={"co2_tco2": "derived from Usage_kWh via emission "
                                   "factor - using it as a feature is target leakage"},
                         notes="15-min industrial energy readings, 2018"),
    "forest_fires": dict(task="regression", target="area", group_col=None,
                         temporal=False, cross_site=False,
                         excluded={}, notes="model on log1p(area); only 52% nonzero; "
                         "temporal split not applicable (month/day only)"),
    "water_potability": dict(task="classification", target="potability",
                             group_col=None, temporal=False, cross_site=False,
                             excluded={}, notes="binary; known label-quality debate - "
                             "discuss in paper"),
    "rain_australia": dict(task="classification", target="raintomorrow",
                           datetime_col="datetime",
                           group_col="location", temporal=True, cross_site=True,
                           excluded={}, notes="already shipped without RISK_MM "
                           "leakage column; ~49 locations 2008-2017"),
}

CATEGORICAL = {
    "beijing_multisite": ["wd", "station"],
    "beijing_pm25": ["cbwd"],
    "air_quality": [],
    "ccpp": [],
    "steel_energy": ["weekstatus", "day_of_week", "load_type"],
    "forest_fires": ["month", "day"],
    "water_potability": [],
    "rain_australia": ["location", "windgustdir", "winddir9am", "winddir3pm",
                       "raintoday"],
}


def report_block(name: str, df: pd.DataFrame, meta: dict, dropped: int) -> str:
    target = meta["target"]
    kept = df
    lines = [f"### {name}", ""]
    lines.append(f"- rows: {len(kept):,} (dropped {dropped:,} with missing target)")
    feats = [c for c in kept.columns if c not in (target, meta.get("datetime_col"))]
    lines.append(f"- features: {len(feats)}")
    miss = kept.isna().mean().drop("datetime", errors="ignore")
    worst = miss[miss > 0].sort_values(ascending=False).head(5)
    if len(worst):
        lines.append("- missingness (top 5): " +
                     ", ".join(f"{c} {v:.0%}" for c, v in worst.items()))
    if meta["task"] == "regression":
        s = kept[target]
        lines.append(f"- target: mean {s.mean():.3f} / median {s.median():.3f} "
                     f"/ std {s.std():.3f}")
    else:
        vc = kept[target].value_counts(normalize=True).sort_index()
        lines.append("- target balance: " +
                     ", ".join(f"{k}={v:.1%}" for k, v in vc.items()))
    if meta.get("datetime_col") and meta["temporal"]:
        lines.append(f"- time range: {kept['datetime'].min()} .. {kept['datetime'].max()}")
    if meta.get("group_col"):
        lines.append(f"- groups ({meta['group_col']}): {kept[meta['group_col']].nunique()}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest, report = {}, ["# Data health report", ""]
    for name, loader in LOADERS.items():
        meta = TASKS[name]
        df = loader()
        target, dtc = meta["target"], meta.get("datetime_col")
        n_before = len(df)
        df = df.dropna(subset=[target]).reset_index(drop=True)
        features = [c for c in df.columns
                    if c not in (target, dtc)
                    and c not in meta.get("excluded", {})]
        df.to_parquet(PROC / f"{name}.parquet", index=False)
        manifest[name] = dict(
            source=SOURCES[name], **meta,
            features=features,
            categorical_features=[c for c in CATEGORICAL[name] if c in features],
            rows=len(df), rows_dropped_missing_target=n_before - len(df),
            processed_file=f"data/processed/{name}.parquet",
        )
        report.append(report_block(name, df, meta, n_before - len(df)))
        print(f"[done] {name}: {len(df):,} rows, {len(features)} features")
    (ROOT / "data" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "data_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\nmanifest + report written")


if __name__ == "__main__":
    main()
