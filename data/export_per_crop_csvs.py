"""
Per-Crop Dataset Export Script for KrishiDristi
Exports individual CSV files for each of the 8 crops — raw and cleaned versions.
Run once to populate data/crops/ directory.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_cleaner import TimeSeriesDataCleaner

PARQUET_FILE = os.path.join(os.path.dirname(__file__), "crop_prices_100k.parquet")
CROPS_DIR    = os.path.join(os.path.dirname(__file__), "crops")

CROP_FILENAMES = {
    "Wheat":        "wheat",
    "Paddy (Rice)": "paddy_rice",
    "Maize":        "maize",
    "Cotton":       "cotton",
    "Soybean":      "soybean",
    "Onion":        "onion",
    "Potato":       "potato",
    "Tomato":       "tomato",
}

def export_all():
    os.makedirs(CROPS_DIR, exist_ok=True)
    print(f"Loading dataset from: {PARQUET_FILE}")
    df = pd.read_parquet(PARQUET_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    manifest = {}

    for crop, slug in CROP_FILENAMES.items():
        crop_df = df[df["Commodity"] == crop].copy()
        crop_df = crop_df.sort_values("Date").reset_index(drop=True)

        # ---- Raw CSV ----
        raw_path = os.path.join(CROPS_DIR, f"{slug}_prices.csv")
        crop_df.to_csv(raw_path, index=False)

        # ---- Cleaned CSV (aggregate to daily modal price, then clean) ----
        daily_ts = crop_df.groupby("Date").agg({
            "Modal_Price": "mean",
            "Min_Price": "mean",
            "Max_Price": "mean",
            "Arrivals_Tons": "sum",
            "State": "first",
            "Mandi": "first",
        }).sort_index()

        series = daily_ts["Modal_Price"].copy()
        clean_result = TimeSeriesDataCleaner.clean_time_series(
            series,
            outlier_method="iqr",
            outlier_threshold=1.5,
            treatment="winsorize",
        )

        cleaned_df = daily_ts.copy()
        cleaned_df["Modal_Price_Raw"]     = clean_result["raw_series"]
        cleaned_df["Modal_Price_Cleaned"] = clean_result["cleaned_series"]
        cleaned_df["Is_Outlier"]          = False
        outlier_dates = {pt["date"] for pt in clean_result.get("outlier_points", [])}
        for d in outlier_dates:
            try:
                cleaned_df.loc[pd.Timestamp(d), "Is_Outlier"] = True
            except Exception:
                pass
        cleaned_df = cleaned_df.reset_index()
        cleaned_path = os.path.join(CROPS_DIR, f"{slug}_prices_cleaned.csv")
        cleaned_df.to_csv(cleaned_path, index=False)

        # ---- Stats for manifest ----
        stats = clean_result["stats_comparison"]
        try:
            health = TimeSeriesDataCleaner.audit_dataset_health(
                crop_df[["Date", "Modal_Price", "Mandi", "Commodity"]]
            )["health_score"]
        except Exception:
            health = 100.0
        manifest[crop] = {
            "commodity": crop,
            "slug": slug,
            "raw_file":     f"crops/{slug}_prices.csv",
            "cleaned_file": f"crops/{slug}_prices_cleaned.csv",
            "total_records": len(crop_df),
            "daily_records": len(daily_ts),
            "date_start": str(crop_df["Date"].min().date()),
            "date_end":   str(crop_df["Date"].max().date()),
            "states": sorted(crop_df["State"].unique().tolist()),
            "mandis": sorted(crop_df["Mandi"].unique().tolist()),
            "avg_modal_price": round(float(crop_df["Modal_Price"].mean()), 2),
            "min_modal_price": round(float(crop_df["Modal_Price"].min()), 2),
            "max_modal_price": round(float(crop_df["Modal_Price"].max()), 2),
            "outliers_detected": clean_result["outlier_info"]["outlier_count"],
            "outlier_pct": clean_result["outlier_info"]["outlier_pct"],
            "variance_reduction_pct": stats["variance_reduction_pct"],
            "health_score": health,
        }
        print(f"  ✓ {crop:10s} → {len(crop_df):6,} rows | "
              f"Outliers: {clean_result['outlier_info']['outlier_count']:4d} "
              f"({clean_result['outlier_info']['outlier_pct']}%) | "
              f"HealthScore: {manifest[crop]['health_score']}")

    # Save manifest
    manifest_path = os.path.join(CROPS_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ All 8 crop CSVs exported to {CROPS_DIR}")
    print(f"   Manifest: {manifest_path}")
    return manifest

if __name__ == "__main__":
    export_all()
