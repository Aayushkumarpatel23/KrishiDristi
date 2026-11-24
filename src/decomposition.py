"""
Time Series Decomposition Module
Implements Additive, Multiplicative, and STL (Seasonal-Trend LOESS) decomposition,
along with quantitative calculations of Trend Strength and Seasonality Strength.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL

class TimeSeriesDecomposer:
    @staticmethod
    def decompose(
        series: pd.Series,
        method: str = "stl", # 'additive', 'multiplicative', or 'stl'
        period: int = 52 # 52 for weekly, 12 for monthly, 365 for daily
    ) -> Dict[str, Any]:
        """
        Decomposes the time series into Trend, Seasonal, and Residual components.
        """
        clean_s = series.dropna()
        n = len(clean_s)
        
        # Guard against period larger than series length
        if n < 2 * period:
            period = max(4, n // 3)
            
        dates = [str(d.date()) if hasattr(d, 'date') else str(d) for d in clean_s.index]
        
        if method.lower() == "stl":
            # STL requires strictly positive odd seasonal length
            seasonal_len = period if period % 2 != 0 else period + 1
            stl_res = STL(clean_s, period=period, seasonal=seasonal_len).fit()
            trend = stl_res.trend
            seasonal = stl_res.seasonal
            residual = stl_res.resid
            model_type = "STL (Loess Robust)"
        elif method.lower() == "multiplicative":
            # Multiplicative requires strictly positive values
            shifted_s = clean_s if (clean_s > 0).all() else (clean_s - clean_s.min() + 1.0)
            res = seasonal_decompose(shifted_s, model="multiplicative", period=period, extrapolate_trend="freq")
            trend = res.trend
            seasonal = res.seasonal
            residual = res.resid
            model_type = "Classical Multiplicative"
        else: # additive
            res = seasonal_decompose(clean_s, model="additive", period=period, extrapolate_trend="freq")
            trend = res.trend
            seasonal = res.seasonal
            residual = res.resid
            model_type = "Classical Additive"

        # Calculate Seasonality Strength (Fs) and Trend Strength (Ft) (Wang, Smith & Hyndman, 2006)
        var_resid = np.nanvar(residual)
        var_detrended = np.nanvar(seasonal + residual)
        var_deseasonalized = np.nanvar(trend + residual)
        
        strength_season = max(0.0, min(1.0, 1.0 - (var_resid / var_detrended))) if var_detrended > 0 else 0.0
        strength_trend = max(0.0, min(1.0, 1.0 - (var_resid / var_deseasonalized))) if var_deseasonalized > 0 else 0.0
        
        return {
            "method": model_type,
            "period": period,
            "dates": dates,
            "observed": clean_s.round(2).tolist(),
            "trend": [None if np.isnan(v) else round(float(v), 2) for v in trend],
            "seasonal": [None if np.isnan(v) else round(float(v), 2) for v in seasonal],
            "residual": [None if np.isnan(v) else round(float(v), 2) for v in residual],
            "metrics": {
                "trend_strength": round(float(strength_trend), 4),
                "seasonality_strength": round(float(strength_season), 4),
                "residual_variance": round(float(var_resid), 2)
            }
        }
