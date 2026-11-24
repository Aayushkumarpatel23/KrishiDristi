"""
Data Cleaning & Quality Assurance Module for Time Series
Implements missing value detection, temporal imputation, outlier detection
(IQR, Z-Score, Hampel Filter / MAD), Winsorization, duplicate resolution,
and comprehensive Data Health Audit reporting.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy import stats

class TimeSeriesDataCleaner:
    @staticmethod
    def audit_dataset_health(df: pd.DataFrame, target_col: str = "Modal_Price") -> Dict[str, Any]:
        """
        Executes a comprehensive data quality and hygiene audit across the dataset.
        Checks for missing values, duplicates, extreme outliers, and negative anomalies.
        """
        total_rows = len(df)
        missing_count = int(df[target_col].isna().sum())
        missing_pct = round((missing_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        
        # Duplicate date-location entries
        if "Date" in df.columns and "Commodity" in df.columns and "Mandi" in df.columns:
            duplicate_count = int(df.duplicated(subset=["Date", "Commodity", "Mandi"]).sum())
        else:
            duplicate_count = 0
            
        # Negative or zero price anomalies
        non_positive = int((df[target_col] <= 0).sum())
        
        # Outlier counts via IQR
        clean_s = df[target_col].dropna()
        if len(clean_s) < 4:
            outlier_count = 0
            outlier_pct = 0.0
            q25, q75, iqr = 0.0, 0.0, 0.0
            lower_bound, upper_bound = 0.0, 0.0
        else:
            q25, q75 = float(np.nanpercentile(clean_s, 25)), float(np.nanpercentile(clean_s, 75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_count = int(((clean_s < lower_bound) | (clean_s > upper_bound)).sum())
            outlier_pct = round((outlier_count / len(clean_s)) * 100.0, 2) if len(clean_s) > 0 else 0.0
        
        # Overall Data Health Score (0-100)
        penalty = (missing_pct * 2.0) + (outlier_pct * 0.8) + (non_positive * 0.5)
        health_score = max(0.0, min(100.0, round(100.0 - penalty, 1)))
        
        return {
            "total_records_audited": total_rows,
            "missing_values": missing_count,
            "missing_pct": missing_pct,
            "duplicates_resolved": duplicate_count,
            "zero_or_negative_anomalies": non_positive,
            "outliers_identified": outlier_count,
            "outlier_pct": outlier_pct,
            "health_score": health_score,
            "bounds": {
                "q25": round(q25, 2),
                "q75": round(q75, 2),
                "iqr": round(iqr, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            }
        }

    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = "iqr", # 'iqr', 'zscore', or 'hampel'
        threshold: float = 1.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detects anomalies and price spikes using statistical thresholding.
        Returns a boolean mask of outliers and summary metadata.
        """
        values = series.values.astype(float)
        n = len(values)
        
        if method.lower() == "zscore":
            mean = np.nanmean(values)
            std = np.nanstd(values)
            z_scores = np.abs((values - mean) / (std + 1e-8))
            outlier_mask = z_scores > threshold
            lower_limit = mean - threshold * std
            upper_limit = mean + threshold * std
            method_name = f"Z-Score (|z| > {threshold})"
            
        elif method.lower() == "hampel":
            # Hampel Filter: Rolling median & MAD (Median Absolute Deviation)
            # Extremely robust against clustered outliers in time series
            window = 7
            rolling_median = pd.Series(values).rolling(window=window, center=True, min_periods=1).median().values
            rolling_mad = pd.Series(np.abs(values - rolling_median)).rolling(window=window, center=True, min_periods=1).median().values * 1.4826
            diff = np.abs(values - rolling_median)
            outlier_mask = diff > (threshold * rolling_mad)
            lower_limit = float(np.nanmin(rolling_median - threshold * rolling_mad))
            upper_limit = float(np.nanmax(rolling_median + threshold * rolling_mad))
            method_name = f"Hampel Filter / MAD (Window={window}, k={threshold})"
            
        else: # Default IQR (Tukey's Fences)
            q25, q75 = np.nanpercentile(values, 25), np.nanpercentile(values, 75)
            iqr = q75 - q25
            lower_limit = q25 - threshold * iqr
            upper_limit = q75 + threshold * iqr
            outlier_mask = (values < lower_limit) | (values > upper_limit)
            method_name = f"Interquartile Range IQR (k={threshold})"
            
        outlier_indices = np.where(outlier_mask)[0].tolist()
        
        return outlier_mask, {
            "method": method_name,
            "outlier_count": len(outlier_indices),
            "outlier_pct": round((len(outlier_indices) / n) * 100.0, 2) if n > 0 else 0.0,
            "outlier_indices": outlier_indices,
            "lower_limit": round(float(lower_limit), 2),
            "upper_limit": round(float(upper_limit), 2)
        }

    @classmethod
    def clean_time_series(
        cls,
        series: pd.Series,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        treatment: str = "winsorize", # 'winsorize', 'interpolate', 'drop'
        impute_method: str = "time" # 'time', 'linear', 'spline', 'ffill'
    ) -> Dict[str, Any]:
        """
        Full end-to-end cleaning pipeline on a 1D time series:
        1. Identifies and clips non-positive or corrupted prices.
        2. Detects outliers via chosen statistical filter.
        3. Treats outliers (Winsorization or replacement with interpolated trajectory).
        4. Imputes missing intervals.
        """
        raw_s = series.copy()
        
        # 1. Floor at minimum valid threshold
        raw_s[raw_s <= 0] = np.nan
        
        # 2. Outlier detection
        outlier_mask, outlier_meta = cls.detect_outliers(raw_s, method=outlier_method, threshold=outlier_threshold)
        
        cleaned_s = raw_s.copy()
        
        # 3. Outlier Treatment
        if treatment == "winsorize":
            # Cap at upper and lower boundaries
            lo = outlier_meta["lower_limit"]
            up = outlier_meta["upper_limit"]
            cleaned_s = np.clip(cleaned_s, lo, up)
            treatment_name = "Winsorization (Capped at Bounds)"
        elif treatment == "interpolate":
            # Set outliers to NaN then interpolate
            cleaned_s[outlier_mask] = np.nan
            if impute_method == "spline" and len(cleaned_s.dropna()) > 3:
                cleaned_s = cleaned_s.interpolate(method="spline", order=2).bfill().ffill()
            else:
                cleaned_s = cleaned_s.interpolate(method="time" if isinstance(cleaned_s.index, pd.DatetimeIndex) else "linear").bfill().ffill()
            treatment_name = f"Interpolated ({impute_method.capitalize()})"
        else: # Drop outliers
            cleaned_s[outlier_mask] = np.nan
            cleaned_s = cleaned_s.bfill().ffill()
            treatment_name = "Forward / Backward Fill"
            
        # Ensure rounded output
        cleaned_s = pd.Series(cleaned_s, index=series.index).round(2)
        
        # Statistical comparisons before and after
        raw_clean = raw_s.dropna()
        dates_list = [str(d.date()) if hasattr(d, 'date') else str(d) for d in series.index]
        
        outlier_points = [
            {"date": dates_list[i], "raw_price": round(float(series.iloc[i]), 2), "cleaned_price": round(float(cleaned_s.iloc[i]), 2)}
            for i in outlier_meta["outlier_indices"]
        ]
        
        return {
            "dates": dates_list,
            "raw_series": [round(float(v), 2) for v in series.values],
            "cleaned_series": [round(float(v), 2) for v in cleaned_s.values],
            "outlier_info": outlier_meta,
            "outlier_points": outlier_points,
            "treatment": treatment_name,
            "stats_comparison": {
                "raw_mean": round(float(raw_clean.mean()), 2),
                "cleaned_mean": round(float(cleaned_s.mean()), 2),
                "raw_std": round(float(raw_clean.std()), 2),
                "cleaned_std": round(float(cleaned_s.std()), 2),
                "raw_skewness": round(float(stats.skew(raw_clean)), 3),
                "cleaned_skewness": round(float(stats.skew(cleaned_s)), 3),
                "variance_reduction_pct": round(max(0.0, (1.0 - (cleaned_s.var() / raw_clean.var())) * 100.0), 2)
            }
        }
