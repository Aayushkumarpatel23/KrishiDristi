"""
Unified Model Manager and Multi-Model Benchmarking Suite
Coordinates training, validation against test split, metrics computation,
and future out-of-sample projection.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

from src.models.baseline import BaselineForecaster
from src.models.arima_sarima import ArimaSarimaForecaster
from src.models.ml_forecaster import MLForecaster
from src.evaluator import TimeSeriesEvaluator
from src.diagnostics import TimeSeriesDiagnostics

class ModelManager:
    @classmethod
    def train_and_evaluate(
        cls,
        model_type: str,
        train_series: pd.Series,
        test_series: pd.Series,
        forecast_horizon: int = 12,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Trains model on train_series, evaluates accuracy on test_series,
        and generates out-of-sample forecast for future horizons.
        """
        params = params or {}
        test_len = len(test_series)
        evaluator = TimeSeriesEvaluator()
        
        # 1. Dispatch Model
        if model_type.lower() == "arima":
            p = int(params.get("p", 1))
            d = int(params.get("d", 1))
            q = int(params.get("q", 1))
            # Test set forecast
            res_test = ArimaSarimaForecaster.fit_and_forecast(
                train_series, steps=test_len, order=(p, d, q), seasonal_order=(0, 0, 0, 0)
            )
            test_preds = np.array(res_test["forecast"])
            
            # Future forecast: Fit on full series (train + test)
            full_series = pd.concat([train_series, test_series])
            res_future = ArimaSarimaForecaster.fit_and_forecast(
                full_series, steps=forecast_horizon, order=(p, d, q), seasonal_order=(0, 0, 0, 0)
            )
            future_preds = res_future["forecast"]
            conf_lower = res_future["lower_bound"]
            conf_upper = res_future["upper_bound"]
            aic = res_future["aic"]
            bic = res_future["bic"]
            model_display_name = f"ARIMA({p},{d},{q})"
            residuals = res_test["residuals"]

        elif model_type.lower() == "sarima":
            p = int(params.get("p", 1))
            d = int(params.get("d", 1))
            q = int(params.get("q", 1))
            P = int(params.get("P", 1))
            D = int(params.get("D", 1))
            Q = int(params.get("Q", 1))
            s = int(params.get("s", 52))
            
            res_test = ArimaSarimaForecaster.fit_and_forecast(
                train_series, steps=test_len, order=(p, d, q), seasonal_order=(P, D, Q, s)
            )
            test_preds = np.array(res_test["forecast"])
            
            full_series = pd.concat([train_series, test_series])
            res_future = ArimaSarimaForecaster.fit_and_forecast(
                full_series, steps=forecast_horizon, order=(p, d, q), seasonal_order=(P, D, Q, s)
            )
            future_preds = res_future["forecast"]
            conf_lower = res_future["lower_bound"]
            conf_upper = res_future["upper_bound"]
            aic = res_future["aic"]
            bic = res_future["bic"]
            model_display_name = f"SARIMA({p},{d},{q})x({P},{D},{Q})_{s}"
            residuals = res_test["residuals"]

        elif model_type.lower() == "holt_winters":
            trend = params.get("trend", "add")
            seasonal = params.get("seasonal", "add")
            s = int(params.get("s", 52))
            
            _, test_preds, _ = BaselineForecaster.holt_winters(
                train_series, steps=test_len, trend=trend, seasonal=seasonal, seasonal_periods=s
            )
            full_series = pd.concat([train_series, test_series])
            _, future_preds_arr, hw_info = BaselineForecaster.holt_winters(
                full_series, steps=forecast_horizon, trend=trend, seasonal=seasonal, seasonal_periods=s
            )
            future_preds = [round(float(v), 2) for v in future_preds_arr]
            # Standard error heuristic for HW confidence bands
            sample_std = float(train_series.std())
            conf_lower = [round(float(v - 1.96 * sample_std * 0.35), 2) for v in future_preds]
            conf_upper = [round(float(v + 1.96 * sample_std * 0.35), 2) for v in future_preds]
            aic = hw_info["aic"]
            bic = hw_info["bic"]
            model_display_name = f"Holt-Winters Seasonal ({trend}, {seasonal})"
            residuals = (test_series.values - test_preds).tolist()

        elif model_type.lower() == "random_forest":
            n_estimators = int(params.get("n_estimators", 80))
            _, test_preds, _ = MLForecaster.fit_and_forecast(
                train_series, steps=test_len, model_type="random_forest", n_estimators=n_estimators
            )
            full_series = pd.concat([train_series, test_series])
            _, future_preds_arr, ml_info = MLForecaster.fit_and_forecast(
                full_series, steps=forecast_horizon, model_type="random_forest", n_estimators=n_estimators
            )
            future_preds = [round(float(v), 2) for v in future_preds_arr]
            sample_std = float(train_series.std())
            conf_lower = [round(float(v - 1.96 * sample_std * 0.4), 2) for v in future_preds]
            conf_upper = [round(float(v + 1.96 * sample_std * 0.4), 2) for v in future_preds]
            aic, bic = None, None
            model_display_name = f"Random Forest Regressor (Lags)"
            residuals = (test_series.values - test_preds).tolist()

        elif model_type.lower() == "moving_average":
            w = int(params.get("window", 4))
            test_preds = BaselineForecaster.moving_average_forecast(train_series, steps=test_len, window=w)
            full_series = pd.concat([train_series, test_series])
            future_preds = [round(float(v), 2) for v in BaselineForecaster.moving_average_forecast(full_series, steps=forecast_horizon, window=w)]
            conf_lower = [round(float(v * 0.92), 2) for v in future_preds]
            conf_upper = [round(float(v * 1.08), 2) for v in future_preds]
            aic, bic = None, None
            model_display_name = f"Moving Average (MA-{w})"
            residuals = (test_series.values - test_preds).tolist()

        else: # Naive
            test_preds = BaselineForecaster.naive_forecast(train_series, steps=test_len)
            full_series = pd.concat([train_series, test_series])
            future_preds = [round(float(v), 2) for v in BaselineForecaster.naive_forecast(full_series, steps=forecast_horizon)]
            conf_lower = [round(float(v * 0.90), 2) for v in future_preds]
            conf_upper = [round(float(v * 1.10), 2) for v in future_preds]
            aic, bic = None, None
            model_display_name = "Naive Baseline"
            residuals = (test_series.values - test_preds).tolist()

        # 2. Compute Test Metrics
        metrics = evaluator.calculate_metrics(test_series.values, test_preds)
        
        # 3. Residual diagnostics
        diag = TimeSeriesDiagnostics.evaluate_residuals(np.array(residuals))

        return {
            "model_type": model_type,
            "model_name": model_display_name,
            "metrics": metrics,
            "aic": aic,
            "bic": bic,
            "test_actual": [round(float(v), 2) for v in test_series.values],
            "test_predicted": [round(float(v), 2) for v in test_preds],
            "future_forecast": future_preds,
            "future_conf_lower": conf_lower,
            "future_conf_upper": conf_upper,
            "residuals": [round(float(v), 2) for v in residuals],
            "diagnostics": diag
        }

    @classmethod
    def benchmark_all_models(
        cls,
        train_series: pd.Series,
        test_series: pd.Series,
        seasonal_period: int = 52
    ) -> List[Dict[str, Any]]:
        """Runs side-by-side comparative benchmarking across all time series models."""
        models_to_test = [
            ("sarima", {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": min(seasonal_period, 12)}),
            ("arima", {"p": 1, "d": 1, "q": 1}),
            ("holt_winters", {"trend": "add", "seasonal": "add", "s": min(seasonal_period, 12)}),
            ("random_forest", {"n_estimators": 60}),
            ("moving_average", {"window": 4}),
            ("naive", {})
        ]
        
        leaderboard = []
        for m_type, p_dict in models_to_test:
            try:
                res = cls.train_and_evaluate(m_type, train_series, test_series, forecast_horizon=8, params=p_dict)
                m = res["metrics"]
                leaderboard.append({
                    "model_type": m_type,
                    "model_name": res["model_name"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "r2": m["r2"],
                    "accuracy_pct": m["accuracy_score_pct"],
                    "aic": res["aic"],
                    "bic": res["bic"],
                    "is_white_noise": res["diagnostics"].get("ljung_box", {}).get("is_white_noise", False)
                })
            except Exception as e:
                continue
                
        # Sort by RMSE ascending (best model first)
        leaderboard.sort(key=lambda x: x["rmse"])
        # Add rank
        for idx, row in enumerate(leaderboard):
            row["rank"] = idx + 1
            
        return leaderboard
