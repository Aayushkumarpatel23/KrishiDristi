"""
Baseline Forecasting Models
Includes Naive, Moving Average, and Holt-Winters Exponential Smoothing.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class BaselineForecaster:
    @staticmethod
    def naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
        """Projects the last known observation forward."""
        last_val = train.iloc[-1]
        return np.full(steps, last_val)

    @staticmethod
    def moving_average_forecast(train: pd.Series, steps: int, window: int = 4) -> np.ndarray:
        """Projects the mean of the last 'window' observations forward."""
        mean_val = train.iloc[-window:].mean()
        return np.full(steps, mean_val)

    @staticmethod
    def holt_winters(
        train: pd.Series,
        steps: int,
        trend: str = "add",
        seasonal: str = "add",
        seasonal_periods: int = 52
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fits Holt-Winters Triple Exponential Smoothing.
        Returns (in_sample_fitted, forecast_values, model_info).
        """
        # Ensure series is positive if multiplicative
        series = train.copy()
        if seasonal == "mul" and (series <= 0).any():
            seasonal = "add"
            
        n = len(series)
        if n < 2 * seasonal_periods:
            seasonal_periods = max(4, n // 3)
            
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
        fitted = model.fit(optimized=True)
        
        in_sample = fitted.fittedvalues.values
        forecast = fitted.forecast(steps).values
        
        info = {
            "model_name": f"Holt-Winters (trend={trend}, seasonal={seasonal}, s={seasonal_periods})",
            "aic": round(float(fitted.aic), 2) if hasattr(fitted, "aic") and not np.isnan(fitted.aic) else None,
            "bic": round(float(fitted.bic), 2) if hasattr(fitted, "bic") and not np.isnan(fitted.bic) else None,
            "params": {
                "alpha": round(float(fitted.params.get("smoothing_level", 0.0)), 4),
                "beta": round(float(fitted.params.get("smoothing_trend", 0.0)), 4),
                "gamma": round(float(fitted.params.get("smoothing_seasonal", 0.0)), 4)
            }
        }
        return in_sample, forecast, info
