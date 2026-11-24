"""
Stationarity Analysis Module
Implements Augmented Dickey-Fuller (ADF) Test, KPSS Test,
Rolling Statistics, Differencing Transformations, and Joint Academic Inference.
"""

import warnings
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
warnings.filterwarnings("ignore")

class StationarityAnalyzer:
    @staticmethod
    def adf_test(series: pd.Series, maxlag: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs Augmented Dickey-Fuller Test.
        H0: The series has a unit root (is non-stationary).
        H1: The series is stationary (no unit root).
        """
        clean_s = series.dropna()
        if len(clean_s) < 10:
            raise ValueError("Time series too short for ADF test (<10 observations).")
            
        result = adfuller(clean_s, maxlag=maxlag, autolag="AIC")
        test_stat = float(result[0])
        p_value = float(result[1])
        used_lag = int(result[2])
        n_obs = int(result[3])
        crit_vals = {k: float(v) for k, v in result[4].items()}
        
        is_stationary = bool(p_value < 0.05)
        
        if is_stationary:
            conclusion = (
                f"Stationary (Reject H0 at p={p_value:.4f} < 0.05). "
                f"Test statistic ({test_stat:.3f}) is more negative than 5% critical value ({crit_vals['5%']:.3f})."
            )
        else:
            conclusion = (
                f"Non-Stationary (Fail to reject H0 at p={p_value:.4f} >= 0.05). "
                f"Unit root present. Differencing (d >= 1) is required."
            )
            
        return {
            "test_name": "Augmented Dickey-Fuller (ADF)",
            "test_statistic": round(test_stat, 4),
            "p_value": round(p_value, 4),
            "lags_used": used_lag,
            "observations": n_obs,
            "critical_values": {k: round(v, 4) for k, v in crit_vals.items()},
            "is_stationary": is_stationary,
            "conclusion": conclusion
        }

    @staticmethod
    def kpss_test(series: pd.Series, regression: str = "c") -> Dict[str, Any]:
        """
        Performs KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test.
        H0: The process is trend/level stationary.
        H1: The process has a unit root (is non-stationary).
        """
        clean_s = series.dropna()
        if len(clean_s) < 10:
            raise ValueError("Time series too short for KPSS test.")
            
        # nlags='auto' is standard in statsmodels
        result = kpss(clean_s, regression=regression, nlags="auto")
        test_stat = float(result[0])
        p_value = float(result[1])
        lags = int(result[2])
        crit_vals = {k: float(v) for k, v in result[3].items()}
        
        # In KPSS, rejecting H0 (p < 0.05) means non-stationary!
        is_stationary = bool(p_value >= 0.05)
        
        if is_stationary:
            conclusion = (
                f"Stationary (Fail to reject H0 at p={p_value:.4f} >= 0.05). "
                f"No evidence of unit root or stochastic drift."
            )
        else:
            conclusion = (
                f"Non-Stationary (Reject H0 at p={p_value:.4f} < 0.05). "
                f"Evidence of unit root / stochastic trend."
            )
            
        return {
            "test_name": "KPSS Test",
            "test_statistic": round(test_stat, 4),
            "p_value": round(p_value, 4),
            "lags_used": lags,
            "critical_values": {k: round(v, 4) for k, v in crit_vals.items()},
            "is_stationary": is_stationary,
            "conclusion": conclusion
        }

    @classmethod
    def comprehensive_stationarity_check(cls, series: pd.Series) -> Dict[str, Any]:
        """
        Combines ADF and KPSS tests to construct the standard 4-quadrant econometric diagnostic matrix.
        """
        adf_res = cls.adf_test(series)
        kpss_res = cls.kpss_test(series)
        
        adf_stat = adf_res["is_stationary"]
        kpss_stat = kpss_res["is_stationary"]
        
        if adf_stat and kpss_stat:
            verdict = "Strictly Stationary"
            recommendation = "Series is ready for ARMA modeling without differencing (d=0)."
            color_badge = "success"
        elif not adf_stat and not kpss_stat:
            verdict = "Non-Stationary (Unit Root)"
            recommendation = "Series possesses a unit root. First differencing (d=1) is necessary before ARIMA fitting."
            color_badge = "danger"
        elif adf_stat and not kpss_stat:
            verdict = "Difference Stationary"
            recommendation = "Series exhibits persistence with weak stationarity. Apply 1st differencing (d=1) for ARIMA."
            color_badge = "warning"
        else: # not adf_stat and kpss_stat
            verdict = "Trend Stationary"
            recommendation = "Series is stationary around a deterministic trend. De-trending or differencing is suggested."
            color_badge = "info"
            
        return {
            "adf": adf_res,
            "kpss": kpss_res,
            "joint_verdict": verdict,
            "recommendation": recommendation,
            "badge": color_badge
        }

    @staticmethod
    def compute_rolling_stats(series: pd.Series, window: int = 12) -> Dict[str, Any]:
        """Calculates rolling mean and rolling standard deviation."""
        clean_s = series.dropna()
        rolling_mean = clean_s.rolling(window=window).mean().dropna()
        rolling_std = clean_s.rolling(window=window).std().dropna()
        
        return {
            "window": window,
            "dates": [str(d.date()) if hasattr(d, 'date') else str(d) for d in rolling_mean.index],
            "original": clean_s.loc[rolling_mean.index].round(2).tolist(),
            "rolling_mean": rolling_mean.round(2).tolist(),
            "rolling_std": rolling_std.round(2).tolist()
        }

    @staticmethod
    def apply_differencing(series: pd.Series, order: int = 1, seasonal_lag: Optional[int] = None) -> pd.Series:
        """Applies regular or seasonal differencing."""
        diff_s = series.copy()
        if order > 0:
            for _ in range(order):
                diff_s = diff_s.diff()
        if seasonal_lag and seasonal_lag > 0:
            diff_s = diff_s.diff(seasonal_lag)
        return diff_s.dropna()
