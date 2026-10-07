"""Download the 8 public datasets for the environmental benchmark project.

UCI datasets come from archive.ics.uci.edu static links (URLs verified 2026-10-05).
Kaggle datasets come via kagglehub (anonymous download of public datasets).
Skips any dataset whose target folder already has files, so the script is
safe to re-run. Exits non-zero if any download failed.
"""
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

UCI = {
    "forest_fires": "https://archive.ics.uci.edu/static/public/162/forest+fires.zip",
    "beijing_multisite": "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
    "air_quality": "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
    "ccpp": "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip",
    "beijing_pm25": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "steel_energy": "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip",
}

KAGGLE = {
    "rain_australia": "jsphyg/weather-dataset-rattle-package",
    "water_potability": "adityakadiwal/water-potability",
}


def has_data(d: Path) -> bool:
    return d.exists() and any(p.name != "_src.zip" for p in d.iterdir())


def fetch(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (env-benchmark)"})
    with urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def main() -> None:
    ok, failed = [], []
    for name, url in UCI.items():
        d = RAW / name
        if has_data(d):
            print(f"[skip] {name}")
            ok.append(name)
            continue
        d.mkdir(parents=True, exist_ok=True)
        zp = d / "_src.zip"
        print(f"[get ] {name} ...", flush=True)
        try:
            fetch(url, zp)
            with zipfile.ZipFile(zp) as z:
                z.extractall(d)
            zp.unlink()
            ok.append(name)
        except Exception as e:  # noqa: BLE001 - report everything, keep going
            failed.append((name, repr(e)))

    for name, handle in KAGGLE.items():
        d = RAW / name
        if has_data(d):
            print(f"[skip] {name}")
            ok.append(name)
            continue
        print(f"[get ] {name} (kagglehub) ...", flush=True)
        try:
            import kagglehub

            p = Path(kagglehub.dataset_download(handle))
            if p.is_dir():
                shutil.copytree(p, d, dirs_exist_ok=True)
            else:
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d / p.name)
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            failed.append((name, repr(e)))

    print("\n=== summary ===")
    for n in ok:
        print("OK  ", n)
    for n, e in failed:
        print("FAIL", n, e)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
