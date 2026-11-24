"""
Time Series Evaluation Metrics Module
Computes MAE, RMSE, MAPE, R2, NRMSE, and Mean Directional Accuracy (MDA).
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class TimeSeriesEvaluator:
    @staticmethod
    def calculate_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Calculates standard econometric and machine learning forecasting accuracy metrics."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        # Remove any mutual NaNs
        valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_t = y_true[valid_mask]
        y_p = y_pred[valid_mask]
        
        if len(y_t) == 0:
            return {"error": "Empty or all-NaN evaluation series."}
            
        mae = float(mean_absolute_error(y_t, y_p))
        mse = float(mean_squared_error(y_t, y_p))
        rmse = float(np.sqrt(mse))
        
        # MAPE (avoiding division by zero)
        non_zero = y_t != 0
        if np.any(non_zero):
            mape = float(np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100.0)
        else:
            mape = 0.0
            
        # R2 Score
        r2 = float(r2_score(y_t, y_p))
        
        # NRMSE (Normalized by mean and by range)
        y_mean = np.mean(y_t)
        y_range = np.ptp(y_t) # max - min
        nrmse_mean = float((rmse / y_mean) * 100.0) if y_mean != 0 else 0.0
        nrmse_range = float((rmse / y_range) * 100.0) if y_range != 0 else 0.0
        
        # Mean Directional Accuracy (MDA): Did the forecast predict the correct up/down direction?
        if len(y_t) > 1:
            true_direction = np.sign(y_t[1:] - y_t[:-1])
            pred_direction = np.sign(y_p[1:] - y_t[:-1])
            mda = float(np.mean(true_direction == pred_direction) * 100.0)
        else:
            mda = 100.0
            
        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
            "nrmse_pct": round(nrmse_mean, 2),
            "mda_pct": round(mda, 2),
            "accuracy_score_pct": round(max(0.0, 100.0 - mape), 2),
            "samples_evaluated": len(y_t)
        }
