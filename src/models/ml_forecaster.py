"""
Machine Learning Time Series Forecaster
Generates autoregressive lag features, rolling windows, and seasonal harmonics
to train Random Forest and Gradient Boosting regression models.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class MLForecaster:
    @staticmethod
    def create_features(series: pd.Series, lags: List[int] = [1, 2, 3, 4, 8, 12, 26, 52]) -> Tuple[pd.DataFrame, pd.Series]:
        """Engineers lag and rolling window features from 1D time series."""
        df = pd.DataFrame({"target": series.values}, index=series.index)
        
        # Lag features
        for lag in lags:
            df[f"lag_{lag}"] = df["target"].shift(lag)
            
        # Rolling statistics
        df["rolling_mean_4"] = df["target"].shift(1).rolling(4).mean()
        df["rolling_std_4"] = df["target"].shift(1).rolling(4).std()
        df["rolling_mean_12"] = df["target"].shift(1).rolling(12).mean()
        
        # Calendar harmonics if datetime index
        if isinstance(series.index, pd.DatetimeIndex):
            day_of_year = series.index.dayofyear if hasattr(series.index, "dayofyear") else getattr(series.index, "day_of_year", None)
            if day_of_year is not None:
                df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
                df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)
                
        df_clean = df.dropna()
        X = df_clean.drop(columns=["target"])
        y = df_clean["target"]
        return X, y

    @classmethod
    def fit_and_forecast(
        cls,
        train: pd.Series,
        steps: int,
        model_type: str = "random_forest",
        n_estimators: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fits ML regressor and performs recursive multi-step forecasting.
        """
        X_train, y_train = cls.create_features(train)
        
        if model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=4, random_state=42)
            name = f"Gradient Boosting (Lags, n={n_estimators})"
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=8, random_state=42, n_jobs=-1)
            name = f"Random Forest Regressor (Lags, n={n_estimators})"
            
        model.fit(X_train, y_train)
        in_sample = model.predict(X_train)
        
        # Feature importance
        feature_importance = dict(zip(X_train.columns, [round(float(v), 4) for v in model.feature_importances_]))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:6])
        
        # Recursive multi-step forecasting
        history = list(train.values)
        forecasts = []
        
        for step in range(steps):
            # Construct feature vector for current step
            feat = {}
            for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
                idx = len(history) - lag
                feat[f"lag_{lag}"] = history[idx] if idx >= 0 else history[0]
                
            last_4 = history[-4:] if len(history) >= 4 else history
            last_12 = history[-12:] if len(history) >= 12 else history
            feat["rolling_mean_4"] = float(np.mean(last_4))
            feat["rolling_std_4"] = float(np.std(last_4)) if len(last_4) > 1 else 0.0
            feat["rolling_mean_12"] = float(np.mean(last_12))
            
            # Dummy seasonality approximation
            feat["sin_doy"] = 0.0
            feat["cos_doy"] = 1.0
            
            # Align features with X_train columns
            feat_df = pd.DataFrame([feat])[X_train.columns]
            pred = float(model.predict(feat_df)[0])
            forecasts.append(pred)
            history.append(pred)
            
        return in_sample, np.array(forecasts), {
            "model_name": name,
            "top_features": feature_importance
        }
