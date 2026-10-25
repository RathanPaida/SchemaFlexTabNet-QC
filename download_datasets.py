"""
Automated Dataset Acquisition and Caching Script for Cross-Cohort Benchmark:
Downloads:
1. Kaggle Heart Disease Dataset (Johnsmith88 duplicate copy)
2. Sulianova Cardiovascular Disease Dataset (70,000 rows)
3. Framingham Heart Study Teaching Subset (4,240 rows)
4. UCI Heart Disease (Cleveland, Hungarian, Switzerland, Long Beach VA - 920 patients)
5. UCI Statlog (Heart) (270 patients)
6. UCI Z-Alizadeh Sani (Tehran Cohort - 303 patients)
"""

import os
import io
import zipfile
import requests
import pandas as pd
from typing import Dict

DATA_RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(DATA_RAW_DIR, exist_ok=True)


def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> bool:
    """Download a file with streaming if not already cached."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[CACHED] {os.path.basename(dest_path)} exists ({os.path.getsize(dest_path):,} bytes).")
        return True

    print(f"[DOWNLOADING] {url} -> {dest_path}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        print(f"[SUCCESS] Saved {os.path.basename(dest_path)} ({os.path.getsize(dest_path):,} bytes).")
        return True
    except Exception as e:
        print(f"[ERROR] Failed downloading {url}: {e}")
        return False


def download_uci_heart_disease() -> bool:
    """Download and extract the 4 UCI Heart Disease cohorts from UCI repository."""
    dest_files = [
        "heart_uci_cleveland.data",
        "heart_uci_hungarian.data",
        "heart_uci_switzerland.data",
        "heart_uci_va.data",
    ]
    all_exist = all(os.path.exists(os.path.join(DATA_RAW_DIR, f)) for f in dest_files)
    if all_exist:
        print(f"[CACHED] UCI Heart Disease 4 cohorts already extracted.")
        return True

    url = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
    print(f"[DOWNLOADING] UCI Heart Disease zip from {url}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        file_map = {
            "processed.cleveland.data": "heart_uci_cleveland.data",
            "processed.hungarian.data": "heart_uci_hungarian.data",
            "processed.switzerland.data": "heart_uci_switzerland.data",
            "processed.va.data": "heart_uci_va.data",
        }
        for zip_name, local_name in file_map.items():
            content = z.read(zip_name)
            out_path = os.path.join(DATA_RAW_DIR, local_name)
            with open(out_path, "wb") as f:
                f.write(content)
            print(f"[SUCCESS] Extracted {local_name} ({len(content):,} bytes).")
        return True
    except Exception as e:
        print(f"[ERROR] UCI Heart Disease download failed: {e}")
        return False


def download_statlog_heart() -> bool:
    """Download UCI Statlog (Heart) dataset."""
    dest_path = os.path.join(DATA_RAW_DIR, "statlog_heart.csv")
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[CACHED] Statlog Heart already exists.")
        return True

    # Try ucimlrepo or fallback
    try:
        from ucimlrepo import fetch_ucirepo
        print("[FETCHING] UCI Statlog Heart via ucimlrepo...")
        statlog = fetch_ucirepo(id=145)
        df = statlog.data.features.copy()
        df["target"] = statlog.data.targets
        df.to_csv(dest_path, index=False)
        print(f"[SUCCESS] Saved statlog_heart.csv ({len(df)} rows).")
        return True
    except Exception as e:
        print(f"[WARN] ucimlrepo statlog fetch failed ({e}), trying static zip...")

    url = "https://archive.ics.uci.edu/static/public/145/statlog+heart.zip"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        data = z.read("heart.dat").decode("utf-8")
        lines = [line.strip().split() for line in data.strip().splitlines()]
        cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        df = pd.DataFrame(lines, columns=cols)
        df.to_csv(dest_path, index=False)
        print(f"[SUCCESS] Saved statlog_heart.csv ({len(df)} rows).")
        return True
    except Exception as e:
        print(f"[ERROR] Statlog download failed: {e}")
        return False


def download_z_alizadeh_sani() -> bool:
    """Download Z-Alizadeh Sani coronary artery disease dataset."""
    dest_path = os.path.join(DATA_RAW_DIR, "z_alizadeh_sani.xlsx")
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[CACHED] Z-Alizadeh Sani already exists.")
        return True

    url = "https://archive.ics.uci.edu/static/public/412/z+alizadeh+sani.zip"
    print(f"[DOWNLOADING] Z-Alizadeh Sani from {url}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        content = z.read("Z-Alizadeh sani dataset.xlsx")
        with open(dest_path, "wb") as f:
            f.write(content)
        print(f"[SUCCESS] Saved z_alizadeh_sani.xlsx ({len(content):,} bytes).")
        return True
    except Exception as e:
        print(f"[ERROR] Z-Alizadeh Sani download failed: {e}")
        return False


def download_all_datasets() -> Dict[str, bool]:
    print("=" * 78)
    print("ACQUIRING CROSS-COHORT BENCHMARK DATASETS")
    print("=" * 78)
    results = {}

    # 1. Kaggle Heart Disease (Johnsmith88 copy - 1,025 rows)
    url_kaggle = "https://raw.githubusercontent.com/maiyakizayad2/csi4142_assignment3_dataset2/main/heart.csv"
    results["kaggle_heart"] = download_file(
        url_kaggle, os.path.join(DATA_RAW_DIR, "heart_kaggle_johnsmith88.csv")
    )

    # 2. Sulianova Cardiovascular Disease (70,000 rows)
    url_cardio = "https://raw.githubusercontent.com/caravanuden/cardio/master/cardio_train.csv"
    results["sulianova_cardio"] = download_file(
        url_cardio, os.path.join(DATA_RAW_DIR, "cardio_sulianova.csv")
    )

    # 3. Framingham Heart Study (4,240 rows)
    url_framingham = "https://raw.githubusercontent.com/JishnuMoorthy/Framingham_Heart_Study/main/Framingham.csv"
    results["framingham"] = download_file(
        url_framingham, os.path.join(DATA_RAW_DIR, "framingham.csv")
    )

    # 4. UCI Heart Disease 4 cohorts (Cleveland, Hungarian, Switzerland, VA)
    results["uci_heart"] = download_uci_heart_disease()

    # 5. UCI Statlog (Heart)
    results["statlog"] = download_statlog_heart()

    # 6. Z-Alizadeh Sani
    results["z_sani"] = download_z_alizadeh_sani()

    print("\nDownload Summary:")
    for k, v in results.items():
        status = "OK" if v else "FAILED"
        print(f"  - {k:<20}: {status}")
    print("=" * 78)
    return results


if __name__ == "__main__":
    download_all_datasets()
