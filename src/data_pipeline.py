"""
Data Pipeline Module for Crop Price Time Series
Handles loading 100K+ records, multi-level filtering, temporal aggregation,
interpolation, outlier smoothing, and strict temporal train/test splitting.
"""

import os
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PARQUET_FILE = os.path.join(DATA_DIR, "crop_prices_100k.parquet")
CSV_FILE = os.path.join(DATA_DIR, "crop_prices_sample.csv")

class CropDataPipeline:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or PARQUET_FILE
        self._df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """Loads dataset from parquet (or fallback CSV) into memory."""
        if self._df is not None:
            return self._df
            
        if os.path.exists(self.filepath):
            self._df = pd.read_parquet(self.filepath)
        elif os.path.exists(CSV_FILE):
            self._df = pd.read_csv(CSV_FILE)
            self._df["Date"] = pd.to_datetime(self._df["Date"])
        else:
            raise FileNotFoundError(f"Dataset not found at {self.filepath}. Run generate_dataset.py first.")
            
        self._df["Date"] = pd.to_datetime(self._df["Date"])
        return self._df
        
    def get_metadata(self) -> Dict[str, Any]:
        """Returns summary stats, unique commodities, states, and mandis."""
        df = self.load_data()
        return {
            "total_records": len(df),
            "date_start": str(df["Date"].min().date()),
            "date_end": str(df["Date"].max().date()),
            "commodities": sorted(df["Commodity"].unique().tolist()),
            "states": sorted(df["State"].unique().tolist()),
            "mandis": sorted(df["Mandi"].unique().tolist()),
            "state_mandi_map": df.groupby("State")["Mandi"].unique().apply(list).to_dict(),
            "commodity_state_map": df.groupby("Commodity")["State"].unique().apply(list).to_dict(),
            "avg_modal_prices": df.groupby("Commodity")["Modal_Price"].mean().round(2).to_dict()
        }

    def filter_and_aggregate(
        self,
        commodity: str,
        state: Optional[str] = None,
        mandi: Optional[str] = None,
        frequency: str = "W", # 'D' for daily, 'W' for weekly, 'M' for monthly
        price_col: str = "Modal_Price",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        apply_cleaning: bool = False,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        cleaning_treatment: str = "winsorize",
    ) -> pd.DataFrame:
        """
        Filters time series for a specific crop and location, resamples to target frequency,
        and ensures continuous regular time indices.
        """
        df = self.load_data()
        mask = (df["Commodity"] == commodity)
        
        if state and state != "All":
            mask &= (df["State"] == state)
        if mandi and mandi != "All":
            mask &= (df["Mandi"] == mandi)
        if start_date:
            mask &= (df["Date"] >= pd.to_datetime(start_date))
        if end_date:
            mask &= (df["Date"] <= pd.to_datetime(end_date))
            
        sub_df = df[mask].sort_values("Date")
        if sub_df.empty:
            raise ValueError(f"No records found for Commodity={commodity}, State={state}, Mandi={mandi}")
            
        # Group by date to average across mandis if "All" is selected
        daily_ts = sub_df.groupby("Date").agg({
            "Modal_Price": "mean",
            "Min_Price": "mean",
            "Max_Price": "mean",
            "Arrivals_Tons": "sum",
            "Rainfall_Anomaly_Pct": "mean",
            "Transport_Fuel_Index": "mean"
        }).sort_index()
        
        # Resample to ensure strictly regular frequency
        freq_mapping = {
            "D": "D",
            "W": "W-MON",
            "M": "MS" # Month start
        }
        pandas_freq = freq_mapping.get(frequency.upper(), "W-MON")
        
        resampled = daily_ts.resample(pandas_freq).agg({
            "Modal_Price": "mean",
            "Min_Price": "mean",
            "Max_Price": "mean",
            "Arrivals_Tons": "sum",
            "Rainfall_Anomaly_Pct": "mean",
            "Transport_Fuel_Index": "mean"
        })
        
        # Interpolate any missing trading intervals (e.g. holiday weeks)
        resampled["Modal_Price"] = resampled["Modal_Price"].interpolate(method="time").bfill().ffill().round(2)
        resampled["Min_Price"] = resampled["Min_Price"].interpolate(method="time").bfill().ffill().round(2)
        resampled["Max_Price"] = resampled["Max_Price"].interpolate(method="time").bfill().ffill().round(2)
        resampled["Arrivals_Tons"] = resampled["Arrivals_Tons"].interpolate(method="linear").bfill().ffill().round(1)
        resampled["Rainfall_Anomaly_Pct"] = resampled["Rainfall_Anomaly_Pct"].interpolate(method="linear").bfill().ffill().round(1)
        resampled["Transport_Fuel_Index"] = resampled["Transport_Fuel_Index"].interpolate(method="linear").bfill().ffill().round(2)
        
        resampled.reset_index(inplace=True)

        # Optional: Apply cleaning pipeline on Modal_Price before returning
        if apply_cleaning:
            try:
                from src.data_cleaner import TimeSeriesDataCleaner
                series = resampled.set_index("Date")["Modal_Price"]
                result = TimeSeriesDataCleaner.clean_time_series(
                    series,
                    outlier_method=outlier_method,
                    outlier_threshold=outlier_threshold,
                    treatment=cleaning_treatment,
                )
                resampled["Modal_Price_Raw"] = result["raw_series"]
                resampled["Modal_Price"] = result["cleaned_series"]
            except Exception:
                pass  # Fail silently — keep original series

        return resampled

    @staticmethod
    def train_test_split(
        df: pd.DataFrame,
        test_ratio: float = 0.15,
        test_periods: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs chronological train/test split without lookahead leakage."""
        n = len(df)
        if test_periods is not None and 0 < test_periods < n:
            split_idx = n - test_periods
        else:
            split_idx = int(n * (1.0 - test_ratio))
            
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        return train, test
