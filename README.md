# 🌾 KrishiDristi: Agricultural Commodity Price Time Series Intelligence Platform
### An End-to-End Econometric & Machine Learning Framework on 100K+ Market Observations

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.15-4B8BBE?style=for-the-badge)](https://www.statsmodels.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)

---

## 📖 Executive Summary
Agricultural commodity markets operate under extreme volatility dictated by seasonal harvest cycles, supply perishability, and unpredictable climatic shocks. **KrishiDristi** is a complete, production-grade time series engineering project developed for the **Time Series Analysis** academic subject.

The system ingests **over 116,000 historical price transaction records (2018–2024)** across multiple commodities (Wheat, Paddy/Rice, Maize, Cotton, Soybean, Onion, Potato, Tomato) and regional APMC Mandis. It executes automated stationarity assessments (ADF & KPSS tests), structural time series decomposition (STL, Additive, Multiplicative), Box-Jenkins model identification (ACF/PACF), parameter tuning, out-of-sample forecasting with 95% confidence bands, and multi-model benchmarking.

All functionality is served via a **high-performance FastAPI backend**, secured with **Gmail & Google Authentication**, and visualized through an **interactive, modern dark-mode analytics dashboard**.

---

## 🏛️ Mathematical Formulations & Theoretical Framework

### 1. Time Series Data Cleaning, Imputation & Anomaly Filtration
Agricultural commodity markets are prone to holiday closures, liquidity gaps, reporting errors, and sudden unseasonal price spikes. KrishiDristi features an automated pre-modeling cleaning pipeline:

#### A. Missing Value Detection & Temporal Imputation
Because time series models (ARIMA, SARIMA) require strictly regular intervals ($t, t+1, \dots$), missing trading days are resolved via **Time-Weighted Spline Interpolation**:
$$Y(t) = a_i + b_i(t - t_i) + c_i(t - t_i)^2 + d_i(t - t_i)^3 \quad \forall t \in [t_i, t_{i+1}]$$
ensuring $C^2$ continuity (continuous 1st and 2nd derivatives) without inducing artificial volatility.

#### B. Statistical Outlier Detection Methods
1. **Interquartile Range (IQR - Tukey's Fences):**
   $$\text{IQR} = Q_3 - Q_1, \quad \text{Valid Range} = [Q_1 - k \cdot \text{IQR}, \; Q_3 + k \cdot \text{IQR}]$$
2. **Standardized Z-Score Filter:**
   $$Z_t = \frac{Y_t - \bar{Y}}{s}, \quad \text{Flag if } |Z_t| > 2.5$$
3. **Hampel Filter (Robust Median Absolute Deviation):**
   $$S_t = 1.4826 \times \text{median}_{i \in W_t}(|Y_i - \tilde{Y}_t|), \quad \text{Flag if } |Y_t - \tilde{Y}_t| > 3 S_t$$

#### C. Outlier Treatment (Winsorization)
To preserve the overall sample size and prevent loss of degrees of freedom while eliminating distortive kurtosis:
$$\hat{Y}_t = \min(\max(Y_t, \text{Lower Limit}), \text{Upper Limit})$$

---

### 2. Weak Stationarity & Dual-Hypothesis Testing
A discrete stochastic process $Y_t$ is weakly stationary if:
$$E[Y_t] = \mu < \infty, \quad \text{Var}(Y_t) = \sigma^2 < \infty, \quad \text{Cov}(Y_t, Y_{t-k}) = \gamma_k$$

#### Augmented Dickey-Fuller (ADF) Test
Tests for the presence of a unit root:
$$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^p \delta_i \Delta Y_{t-i} + \epsilon_t$$
* **$H_0$:** $\gamma = 0$ (Unit root exists $\implies$ Non-stationary)
* **$H_1$:** $\gamma < 0$ (Stationary)

#### KPSS Test (Kwiatkowski et al., 1992)
Tests stationarity against unit root alternative:
$$Y_t = \xi t + r_t + \epsilon_t \quad \text{where } r_t = r_{t-1} + u_t$$
* **$H_0$:** $\sigma_u^2 = 0$ (Series is trend/level stationary)
* **$H_1$:** $\sigma_u^2 > 0$ (Unit root non-stationary)

#### Joint 4-Quadrant Inference Matrix
| ADF Decision ($p < 0.05$) | KPSS Decision ($p \ge 0.05$) | Joint Diagnostic Conclusion | Action Required |
|---|---|---|---|
| Reject $H_0$ (Stationary) | Fail to Reject $H_0$ (Stationary) | **Strictly Stationary** | Ready for ARMA ($d=0$) |
| Fail to Reject $H_0$ (Non-stat) | Reject $H_0$ (Non-stat) | **Unit Root Non-Stationary** | Apply First Difference ($d=1$) |
| Reject $H_0$ (Stationary) | Reject $H_0$ (Non-stat) | **Difference Stationary** | Apply Differencing ($d=1$) |
| Fail to Reject $H_0$ (Non-stat) | Fail to Reject $H_0$ (Stationary) | **Trend Stationary** | Apply De-trending / Drift |

---

### 3. Time Series Structural Decomposition
Using Classical or LOESS (STL) formulations:
$$Y_t = T_t + S_t + R_t \quad (\text{Additive})$$
$$Y_t = T_t \times S_t \times R_t \quad (\text{Multiplicative})$$

Quantitative Trend Strength ($F_T$) and Seasonality Strength ($F_S$) (Wang, Smith & Hyndman, 2006):
$$F_T = \max\left(0, 1 - \frac{\text{Var}(R)}{\text{Var}(T + R)}\right), \quad F_S = \max\left(0, 1 - \frac{\text{Var}(R)}{\text{Var}(S + R)}\right)$$

---

### 4. Forecasting Models

#### A. ARIMA$(p, d, q)$ Model
$$\Phi_p(B) (1 - B)^d Y_t = \Theta_q(B) \epsilon_t$$
Where:
* $B$ is the backshift operator ($B^k Y_t = Y_{t-k}$)
* $\Phi_p(B) = 1 - \phi_1 B - \dots - \phi_p B^p$ (Autoregressive polynomial)
* $\Theta_q(B) = 1 + \theta_1 B + \dots + \theta_q B^q$ (Moving Average polynomial)

#### B. Seasonal ARIMA (SARIMA) $(p, d, q) \times (P, D, Q)_s$
$$\Phi_p(B) \tilde{\Phi}_P(B^s) (1 - B)^d (1 - B^s)^D Y_t = \Theta_q(B) \tilde{\Theta}_Q(B^s) \epsilon_t$$
Where $s$ is the seasonal period ($s=52$ for weekly, $s=12$ for monthly).

#### C. Holt-Winters Triple Exponential Smoothing
* Level: $\ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$
* Trend: $b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$
* Seasonal: $s_t = \gamma(y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)s_{t-m}$
* Forecast: $\hat{y}_{t+h} = \ell_t + h b_t + s_{t-m + h_m^+}$

#### D. Autoregressive Machine Learning Forecaster
Constructs lagged feature matrix $\mathbf{X} = [y_{t-1}, y_{t-2}, y_{t-4}, y_{t-12}, \mu_{t-4}, \sigma_{t-4}]$ to train multi-tree Random Forest and Gradient Boosting ensembles.

---

### 5. Forecasting Accuracy & Evaluation Metrics
$$\text{MAE} = \frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|$$
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2}$$
$$\text{MAPE} = \frac{100\%}{n} \sum_{t=1}^n \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$
$$R^2 = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}$$
$$\text{MDA} = \frac{100\%}{n-1} \sum_{t=2}^n \mathbf{1}_{[\text{sign}(y_t - y_{t-1}) == \text{sign}(\hat{y}_t - y_{t-1})]}$$

---

### 6. Residual Diagnostic Checking
To confirm the model has extracted all systematic structure, residuals $e_t$ must be white noise:
* **Ljung-Box Portmanteau Test:**
  $$Q = n(n+2) \sum_{k=1}^h \frac{\hat{\rho}_k^2}{n - k} \sim \chi^2(h - p - q)$$
* **Jarque-Bera Test:**
  $$JB = \frac{n}{6}\left(S^2 + \frac{(K - 3)^2}{4}\right) \sim \chi^2(2)$$

---

## 📂 Project Directory Structure

```text
KrishiDristi/
├── data/
│   ├── generate_dataset.py          # 100K+ transaction simulation engine
│   ├── crop_prices_100k.parquet     # Compressed high-volume storage (116K rows)
│   ├── crop_prices_sample.csv       # Preview sample CSV
│   └── dataset_metadata.json        # Precomputed summary metadata
├── src/
│   ├── data_pipeline.py             # Ingestion, aggregation, interpolation & split
│   ├── data_cleaner.py              # Missing imputation, outlier detection (IQR/Hampel) & hygiene
│   ├── stationarity.py              # ADF & KPSS dual test engine & differencing
│   ├── decomposition.py             # STL, Additive & Multiplicative components
│   ├── diagnostics.py              # ACF, PACF, Ljung-Box, Jarque-Bera & Q-Q plot
│   ├── evaluator.py                # MAE, RMSE, MAPE, R2, MDA calculations
│   └── models/
│       ├── baseline.py              # Naive, Moving Average & Holt-Winters
│       ├── arima_sarima.py          # SARIMAX, auto-tuning & confidence intervals
│       ├── ml_forecaster.py         # Lag-feature Random Forest & recursive multi-step
│       └── model_manager.py         # Unified training, testing & benchmarking
├── backend/
│   └── app.py                       # FastAPI REST API, Google Auth & static web server
├── frontend/
│   ├── templates/
│   │   └── index.html               # Modern dark-mode dashboard with Gmail sign-in
│   └── static/
│       └── js/
│           └── app.js               # Dynamic controllers, auth state & Chart.js engine
├── notebooks/
│   ├── generate_notebook.py         # Script to compile academic notebook
│   └── crop_price_time_series_walkthrough.ipynb  # Comprehensive academic walkthrough
├── run.py                           # One-click startup runner
└── README.md                        # Academic documentation
```

---

## 🚀 Quick Start Guide

### 1. Launch the Web Platform
From the `KrishiDristi` directory:
```bash
cd C:\Users\aayus\.gemini\antigravity\scratch\KrishiDristi
python run.py
```
* **Interactive Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Login with Gmail ID
* Click **"Login with Gmail"** in the top navigation bar.
* Enter any valid Gmail address (e.g. `your.name@gmail.com`) or click **"Continue with Google"** for one-click access.
* Your session and profile are remembered persistently in your browser.

### 3. Run the Academic Jupyter Notebook
Open the notebook in JupyterLab or VSCode:
```bash
jupyter notebook notebooks/crop_price_time_series_walkthrough.ipynb
```

### 4. Run Verification Tests
```bash
python tests/test_direct.py
```

---

## 🌐 REST API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/login` | Authenticates with Gmail ID / Google identity |
| `GET` | `/api/auth/session` | Validates active session token |
| `POST` | `/api/auth/logout` | Terminates active user session |
| `GET` | `/api/dataset/metadata` | Total record count, date ranges, commodities & mandis |
| `GET` | `/api/data/health` | Comprehensive dataset hygiene & data health audit |
| `POST` | `/api/data/clean` | Outlier detection (IQR, Z-Score, Hampel) & Winsorization |
| `POST` | `/api/dataset/timeseries` | Filtered, aggregated time series with prices & arrivals |
| `POST` | `/api/analysis/stationarity` | ADF & KPSS test statistics, critical values & rolling stats |
| `POST` | `/api/analysis/decomposition`| STL / Additive trend, seasonal & residual arrays |
| `POST` | `/api/analysis/acf_pacf` | Autocorrelation & Partial Autocorrelation with 95% CI |
| `POST` | `/api/models/train` | Fits model, test evaluation & out-of-sample forecast |
| `POST` | `/api/models/auto_tune` | Box-Jenkins grid search with minimum AIC/BIC scoring |
| `POST` | `/api/models/benchmark` | Side-by-side leaderboard ranking all 6 models |
| `POST` | `/api/export/forecast_csv` | Download forecast table with confidence bands as CSV |
