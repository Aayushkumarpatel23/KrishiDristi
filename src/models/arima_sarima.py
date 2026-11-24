"""
ARIMA and SARIMA Time Series Forecasting Module
Uses statsmodels SARIMAX with robust estimation, AIC/BIC tracking,
parameter grid search, and 95% confidence interval forecasting.
"""

import warnings
from typing import Dict, Any, Tuple, Optional, List
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

class ArimaSarimaForecaster:
    @staticmethod
    def fit_and_forecast(
        train: pd.Series,
        steps: int,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        alpha: float = 0.05 # 95% confidence interval
    ) -> Dict[str, Any]:
        """
        Fits ARIMA / SARIMA model on training series and forecasts forward 'steps' periods.
        """
        clean_train = train.dropna()
        p, d, q = order
        P, D, Q, s = seasonal_order
        
        # If s > 0 and series length is small, adapt s
        if s > 0 and len(clean_train) < 2 * s:
            s = max(4, len(clean_train) // 4)
            seasonal_order = (P, D, Q, s)
            
        model = SARIMAX(
            clean_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=80)
        
        # In-sample fitted values and residuals
        in_sample = fitted.fittedvalues
        residuals = fitted.resid
        
        # Out-of-sample forecast with confidence interval
        forecast_res = fitted.get_forecast(steps=steps)
        forecast_mean = forecast_res.predicted_mean.values
        conf_int = forecast_res.conf_int(alpha=alpha).values
        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]
        
        # Coefficient summary
        params = {k: round(float(v), 4) for k, v in fitted.params.items()}
        pvalues = {k: round(float(v), 4) for k, v in fitted.pvalues.items()}
        
        model_name = (
            f"ARIMA({p},{d},{q})"
            if seasonal_order == (0, 0, 0, 0) or s == 0
            else f"SARIMA({p},{d},{q})x({P},{D},{Q})_{s}"
        )
        
        return {
            "model_name": model_name,
            "order": order,
            "seasonal_order": seasonal_order,
            "in_sample": [round(float(v), 2) for v in in_sample],
            "forecast": [round(float(v), 2) for v in forecast_mean],
            "lower_bound": [round(float(v), 2) for v in lower_bound],
            "upper_bound": [round(float(v), 2) for v in upper_bound],
            "residuals": [round(float(v), 2) for v in residuals],
            "aic": round(float(fitted.aic), 2),
            "bic": round(float(fitted.bic), 2),
            "hqic": round(float(fitted.hqic), 2),
            "log_likelihood": round(float(fitted.llf), 2),
            "parameters": params,
            "p_values": pvalues
        }

    @classmethod
    def auto_order_search(
        cls,
        train: pd.Series,
        max_p: int = 2,
        max_d: int = 1,
        max_q: int = 2,
        is_seasonal: bool = False,
        s: int = 52,
        criterion: str = "aic"
    ) -> Dict[str, Any]:
        """
        Performs a rapid Box-Jenkins grid search across (p, d, q) orders
        to determine optimal hyperparameters by minimizing AIC or BIC.
        """
        clean_train = train.dropna()
        p_vals = range(0, max_p + 1)
        d_vals = range(0, max_d + 1)
        q_vals = range(0, max_q + 1)
        
        best_score = float("inf")
        best_order = (1, 1, 1)
        best_seasonal = (0, 0, 0, 0)
        evaluated_models = []
        
        if is_seasonal and s > 0:
            seasonal_candidates = [(0, 0, 0, s), (1, 0, 0, s), (0, 1, 1, s), (1, 1, 1, s)]
        else:
            seasonal_candidates = [(0, 0, 0, 0)]
            
        for p, d, q in itertools.product(p_vals, d_vals, q_vals):
            if p == 0 and q == 0 and d == 0:
                continue
            for P, D, Q, s_val in seasonal_candidates:
                try:
                    mod = SARIMAX(
                        clean_train,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s_val),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    res = mod.fit(disp=False, maxiter=40)
                    score = res.aic if criterion == "aic" else res.bic
                    
                    name = f"ARIMA({p},{d},{q})" if s_val == 0 else f"SARIMA({p},{d},{q})x({P},{D},{Q})_{s_val}"
                    evaluated_models.append({
                        "name": name,
                        "order": (p, d, q),
                        "seasonal_order": (P, D, Q, s_val),
                        "aic": round(float(res.aic), 2),
                        "bic": round(float(res.bic), 2)
                    })
                    
                    if not np.isnan(score) and score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                        best_seasonal = (P, D, Q, s_val)
                except Exception:
                    continue
                    
        # Sort leaderboard by selected criterion
        evaluated_models.sort(key=lambda x: x[criterion])
        
        return {
            "best_order": best_order,
            "best_seasonal_order": best_seasonal,
            "best_score": round(best_score, 2),
            "criterion": criterion,
            "search_history": evaluated_models[:8] # Top 8 candidates
        }
