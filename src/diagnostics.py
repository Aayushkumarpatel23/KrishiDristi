"""
Time Series Diagnostics Module
Calculates Autocorrelation Function (ACF), Partial Autocorrelation (PACF),
Ljung-Box White Noise Portmanteau test, and Jarque-Bera Residual Normality metrics.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

class TimeSeriesDiagnostics:
    @staticmethod
    def calculate_acf_pacf(series: pd.Series, nlags: int = 40) -> Dict[str, Any]:
        """Calculates ACF and PACF along with 95% Bartlett confidence limits."""
        clean_s = series.dropna()
        n = len(clean_s)
        
        # Max lags cannot exceed half the series length
        max_possible_lags = min(nlags, (n // 2) - 1)
        max_possible_lags = max(10, max_possible_lags)
        
        acf_vals, acf_confint = acf(clean_s, nlags=max_possible_lags, alpha=0.05)
        pacf_vals, pacf_confint = pacf(clean_s, nlags=max_possible_lags, alpha=0.05, method="yule_walker")
        
        # 95% Confidence interval band: +/- 1.96 / sqrt(N)
        ci_bound = round(1.96 / np.sqrt(n), 4)
        
        lags = list(range(len(acf_vals)))
        
        return {
            "lags": lags,
            "ci_bound": ci_bound,
            "acf": [round(float(v), 4) for v in acf_vals],
            "acf_lower": [round(float(c[0]), 4) for c in acf_confint],
            "acf_upper": [round(float(c[1]), 4) for c in acf_confint],
            "pacf": [round(float(v), 4) for v in pacf_vals],
            "pacf_lower": [round(float(c[0]), 4) for c in pacf_confint],
            "pacf_upper": [round(float(c[1]), 4) for c in pacf_confint]
        }

    @staticmethod
    def evaluate_residuals(residuals: np.ndarray, lags: int = 10) -> Dict[str, Any]:
        """
        Runs comprehensive residual diagnostic checks:
        1. Ljung-Box test for autocorrelation / white noise.
        2. Jarque-Bera test for residual normality.
        3. Mean, Standard Deviation, Skewness, Kurtosis.
        4. Quantile-Quantile (Q-Q) coordinates.
        """
        res_clean = np.array(residuals)[~np.isnan(residuals)]
        n = len(res_clean)
        if n < 10:
            return {"status": "insufficient_data"}
            
        # 1. Ljung-Box test
        try:
            lb_df = acorr_ljungbox(res_clean, lags=[min(lags, max(1, n // 4))], return_df=True)
            lb_stat = float(lb_df["lb_stat"].iloc[-1])
            lb_pvalue = float(lb_df["lb_pvalue"].iloc[-1])
        except Exception:
            # Fallback if return_df behavior differs
            lb_res = acorr_ljungbox(res_clean, lags=[min(lags, max(1, n // 4))])
            if isinstance(lb_res, pd.DataFrame):
                lb_stat = float(lb_res["lb_stat"].iloc[-1])
                lb_pvalue = float(lb_res["lb_pvalue"].iloc[-1])
            else:
                lb_stat = float(lb_res[0][-1])
                lb_pvalue = float(lb_res[1][-1])
        is_white_noise = bool(lb_pvalue > 0.05)
        
        # 2. Jarque-Bera test
        jb_res = stats.jarque_bera(res_clean)
        jb_stat, jb_pvalue = float(jb_res.statistic), float(jb_res.pvalue)
        skew = float(stats.skew(res_clean))
        kurtosis = float(stats.kurtosis(res_clean))
        is_normal = bool(jb_pvalue > 0.05)
        
        # 3. Q-Q Plot Coordinates
        sorted_res = np.sort(res_clean)
        # Standardize
        std_res = (sorted_res - np.mean(sorted_res)) / (np.std(sorted_res) + 1e-8)
        norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(std_res)))
        
        # Downsample Q-Q points if large for fast UI rendering
        sample_step = max(1, len(std_res) // 100)
        
        return {
            "sample_size": n,
            "mean": round(float(np.mean(res_clean)), 4),
            "std": round(float(np.std(res_clean)), 4),
            "skewness": round(float(skew), 4),
            "kurtosis": round(float(kurtosis), 4),
            "ljung_box": {
                "statistic": round(lb_stat, 4),
                "p_value": round(lb_pvalue, 4),
                "is_white_noise": is_white_noise,
                "interpretation": (
                    "Residuals behave as White Noise (Fail to reject H0, p > 0.05). No autocorrelation left."
                    if is_white_noise else
                    "Residuals exhibit remaining Autocorrelation (Reject H0, p <= 0.05). Model order tuning recommended."
                )
            },
            "jarque_bera": {
                "statistic": round(float(jb_stat), 4),
                "p_value": round(float(jb_pvalue), 4),
                "is_normal": is_normal,
                "interpretation": (
                    "Residuals are Normally Distributed (Fail to reject H0, p > 0.05)."
                    if is_normal else
                    "Residual distribution deviates from pure normality (Reject H0, p <= 0.05)."
                )
            },
            "qq_plot": {
                "theoretical_quantiles": [round(float(q), 3) for q in norm_quantiles[::sample_step]],
                "sample_quantiles": [round(float(s), 3) for s in std_res[::sample_step]]
            }
        }
