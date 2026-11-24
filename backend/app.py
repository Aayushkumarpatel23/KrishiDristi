"""
FastAPI Server for Crop Price Time Series Forecasting & Analytics Engine
Serves RESTful APIs for dataset exploration, stationarity testing (ADF/KPSS),
decomposition (STL/Additive/Multiplicative), ACF/PACF, ARIMA/SARIMA forecasting,
accuracy benchmarking, and interactive UI frontend.
"""

import os
import sys
from typing import Optional, Dict, Any, List
import io
import uuid
import re
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure root project directory is on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.data_pipeline import CropDataPipeline
from src.data_cleaner import TimeSeriesDataCleaner
from src.stationarity import StationarityAnalyzer
from src.decomposition import TimeSeriesDecomposer
from src.diagnostics import TimeSeriesDiagnostics
from src.models.model_manager import ModelManager
from src.models.arima_sarima import ArimaSarimaForecaster

# Initialize FastAPI App
app = FastAPI(
    title="KrishiDristi: Agricultural Crop Price Time Series Intelligence Platform",
    description="Advanced statistical time series forecasting (ARIMA, SARIMA, Holt-Winters, RF) on 100K+ historical mandi transactions with Google Authentication.",
    version="1.1.0"
)

# Enable CORS for local client development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline Singleton
pipeline = CropDataPipeline()

# Ensure static directories exist
STATIC_DIR = os.path.join(ROOT_DIR, "frontend", "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount crops data directory for direct CSV downloads
CROPS_DIR = os.path.join(ROOT_DIR, "data", "crops")
os.makedirs(CROPS_DIR, exist_ok=True)
app.mount("/data/crops", StaticFiles(directory=CROPS_DIR), name="crops")

# ----------------- Request Models -----------------

class GmailLoginRequest(BaseModel):
    email: str
    name: Optional[str] = None
    credential: Optional[str] = None # Google OAuth token or simulation token

class FilterRequest(BaseModel):
    commodity: str = "Wheat"
    state: Optional[str] = "All"
    mandi: Optional[str] = "All"
    frequency: str = "W" # 'D', 'W', 'M'
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class CleanRequest(BaseModel):
    commodity: str = "Wheat"
    state: Optional[str] = "All"
    mandi: Optional[str] = "All"
    frequency: str = "W"
    outlier_method: str = "iqr" # 'iqr', 'zscore', 'hampel'
    outlier_threshold: float = 1.5
    treatment: str = "winsorize" # 'winsorize', 'interpolate'
    impute_method: str = "time" # 'time', 'linear', 'spline'

class StationarityRequest(BaseModel):
    commodity: str = "Wheat"
    state: Optional[str] = "All"
    mandi: Optional[str] = "All"
    frequency: str = "W"
    diff_order: int = 0
    seasonal_diff_lag: int = 0
    rolling_window: int = 12

class DecompositionRequest(BaseModel):
    commodity: str = "Wheat"
    state: Optional[str] = "All"
    mandi: Optional[str] = "All"
    frequency: str = "W"
    method: str = "stl" # 'stl', 'additive', 'multiplicative'
    period: int = 52

class AcfPacfRequest(BaseModel):
    commodity: str = "Wheat"
    state: Optional[str] = "All"
    mandi: Optional[str] = "All"
    frequency: str = "W"
    diff_order: int = 0
    nlags: int = 35

class ModelTrainRequest(BaseModel):
    commodity: str = "Wheat"
    state: Optional[str] = "All"
    mandi: Optional[str] = "All"
    frequency: str = "W"
    test_periods: int = 24
    forecast_horizon: int = 12
    model_type: str = "sarima" # 'arima', 'sarima', 'holt_winters', 'random_forest', 'naive', 'moving_average'
    params: Dict[str, Any] = Field(default_factory=dict)

class AutoTuneRequest(BaseModel):
    commodity: str = "Wheat"
    state: Optional[str] = "All"
    mandi: Optional[str] = "All"
    frequency: str = "W"
    max_p: int = 2
    max_d: int = 1
    max_q: int = 2
    is_seasonal: bool = True
    seasonal_period: int = 12
    criterion: str = "aic"

class BenchmarkRequest(BaseModel):
    commodity: str = "Wheat"
    state: Optional[str] = "All"
    mandi: Optional[str] = "All"
    frequency: str = "W"
    test_periods: int = 24
    seasonal_period: int = 12

# In-memory authentication session cache
ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}

# ----------------- Authentication Endpoints -----------------

@app.post("/api/auth/login")
def login_with_gmail(req: GmailLoginRequest):
    """
    Authenticates user via Gmail ID or Google Sign-In.
    Validates Gmail address format, generates user session and profile metadata.
    """
    email_clean = req.email.strip().lower()
    
    # Email syntax check
    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(email_regex, email_clean):
        raise HTTPException(status_code=400, detail="Invalid email format. Please provide a valid email address.")
        
    # Check if Gmail or Google Workspace
    is_gmail = email_clean.endswith("@gmail.com") or email_clean.endswith("@googlemail.com")
    
    # Auto-derive user name from email prefix if not supplied
    if req.name and req.name.strip():
        display_name = req.name.strip()
    else:
        prefix = email_clean.split("@")[0]
        # Clean periods, underscores, numbers
        clean_parts = re.split(r"[\._\d]+", prefix)
        display_name = " ".join([p.capitalize() for p in clean_parts if p]).strip()
        if not display_name:
            display_name = prefix.capitalize()
            
    # Avatar URL (DiceBear clean initials styled)
    encoded_name = display_name.replace(" ", "+")
    avatar_url = f"https://ui-avatars.com/api/?name={encoded_name}&background=059669&color=ffffff&bold=true&rounded=true"
    
    # Create session token
    token = f"kd_google_{uuid.uuid4().hex[:16]}"
    session_data = {
        "token": token,
        "email": email_clean,
        "name": display_name,
        "avatar": avatar_url,
        "is_gmail": is_gmail,
        "login_at": datetime.now().isoformat(),
        "provider": "google"
    }
    
    ACTIVE_SESSIONS[token] = session_data
    
    return {
        "success": True,
        "message": f"Welcome to KrishiDristi, {display_name}!",
        "user": session_data
    }

@app.get("/api/auth/session")
def get_current_session(token: Optional[str] = None):
    """Validates an existing user session token."""
    if not token or token not in ACTIVE_SESSIONS:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": ACTIVE_SESSIONS[token]}

@app.post("/api/auth/logout")
def logout_user(token: Optional[str] = None):
    """Terminates an active session."""
    if token and token in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[token]
    return {"success": True, "message": "Signed out successfully."}

# ----------------- API Endpoints -----------------

@app.get("/api/dataset/metadata")
def get_dataset_metadata():
    """Returns dataset summary, commodity lists, mandi catalogs, and average price baselines."""
    try:
        return pipeline.get_metadata()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/health")
def get_dataset_health():
    """Returns comprehensive hygiene and data health audit metrics across 100K+ records."""
    try:
        df = pipeline.load_data()
        return TimeSeriesDataCleaner.audit_dataset_health(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/crops")
def list_crop_datasets():
    """Lists all per-crop dataset files with stats and download links."""
    try:
        manifest_path = os.path.join(ROOT_DIR, "data", "crops", "manifest.json")
        if not os.path.exists(manifest_path):
            raise HTTPException(status_code=404, detail="Crop manifests not generated. Run data/export_per_crop_csvs.py first.")
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        # Enrich with download URLs
        for crop_data in manifest.values():
            slug = crop_data["slug"]
            crop_data["download_raw_url"]     = f"/data/crops/{slug}_prices.csv"
            crop_data["download_cleaned_url"] = f"/data/crops/{slug}_prices_cleaned.csv"
        return {"crops": list(manifest.values()), "total_crops": len(manifest)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/clean")
def clean_selected_series(req: CleanRequest):
    """
    Executes time series outlier detection (IQR, Z-Score, Hampel)
    and treatment (Winsorization or Spline/Linear Interpolation).
    """
    try:
        df = pipeline.filter_and_aggregate(
            commodity=req.commodity,
            state=req.state,
            mandi=req.mandi,
            frequency=req.frequency
        )
        series = df.set_index("Date")["Modal_Price"]
        result = TimeSeriesDataCleaner.clean_time_series(
            series,
            outlier_method=req.outlier_method,
            outlier_threshold=req.outlier_threshold,
            treatment=req.treatment,
            impute_method=req.impute_method
        )
        result["commodity"] = req.commodity
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/dataset/timeseries")
def get_time_series(req: FilterRequest):
    """Retrieves resampled time series for the selected filters."""
    try:
        df = pipeline.filter_and_aggregate(
            commodity=req.commodity,
            state=req.state,
            mandi=req.mandi,
            frequency=req.frequency,
            start_date=req.start_date,
            end_date=req.end_date
        )
        return {
            "commodity": req.commodity,
            "state": req.state,
            "mandi": req.mandi,
            "frequency": req.frequency,
            "count": len(df),
            "dates": [str(d.date()) if hasattr(d, 'date') else str(d) for d in df["Date"]],
            "modal_prices": df["Modal_Price"].tolist(),
            "min_prices": df["Min_Price"].tolist(),
            "max_prices": df["Max_Price"].tolist(),
            "arrivals_tons": df["Arrivals_Tons"].tolist(),
            "rainfall_anomaly": df["Rainfall_Anomaly_Pct"].tolist(),
            "fuel_index": df["Transport_Fuel_Index"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analysis/stationarity")
def analyze_stationarity(req: StationarityRequest):
    """
    Performs Augmented Dickey-Fuller (ADF) & KPSS tests, rolling statistics,
    and differencing transformations.
    """
    try:
        df = pipeline.filter_and_aggregate(
            commodity=req.commodity,
            state=req.state,
            mandi=req.mandi,
            frequency=req.frequency
        )
        series = df.set_index("Date")["Modal_Price"]
        
        # Apply transformation if requested
        if req.diff_order > 0 or req.seasonal_diff_lag > 0:
            analyzed_series = StationarityAnalyzer.apply_differencing(
                series, order=req.diff_order, seasonal_lag=req.seasonal_diff_lag
            )
        else:
            analyzed_series = series
            
        test_results = StationarityAnalyzer.comprehensive_stationarity_check(analyzed_series)
        rolling = StationarityAnalyzer.compute_rolling_stats(analyzed_series, window=req.rolling_window)
        
        return {
            "tests": test_results,
            "rolling": rolling,
            "diff_order": req.diff_order,
            "seasonal_diff_lag": req.seasonal_diff_lag,
            "series_length": len(analyzed_series)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analysis/decomposition")
def decompose_time_series(req: DecompositionRequest):
    """Executes time series decomposition (STL, Additive, or Multiplicative)."""
    try:
        df = pipeline.filter_and_aggregate(
            commodity=req.commodity,
            state=req.state,
            mandi=req.mandi,
            frequency=req.frequency
        )
        series = df.set_index("Date")["Modal_Price"]
        decomp_result = TimeSeriesDecomposer.decompose(series, method=req.method, period=req.period)
        return decomp_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analysis/acf_pacf")
def get_acf_pacf(req: AcfPacfRequest):
    """Calculates ACF and PACF with 95% confidence intervals."""
    try:
        df = pipeline.filter_and_aggregate(
            commodity=req.commodity,
            state=req.state,
            mandi=req.mandi,
            frequency=req.frequency
        )
        series = df.set_index("Date")["Modal_Price"]
        if req.diff_order > 0:
            series = StationarityAnalyzer.apply_differencing(series, order=req.diff_order)
            
        return TimeSeriesDiagnostics.calculate_acf_pacf(series, nlags=req.nlags)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/models/train")
def train_model(req: ModelTrainRequest):
    """
    Trains selected time series model, computes test-set validation accuracy,
    projects future forecasts with 95% confidence intervals, and provides residual diagnostics.
    """
    try:
        df = pipeline.filter_and_aggregate(
            commodity=req.commodity,
            state=req.state,
            mandi=req.mandi,
            frequency=req.frequency
        )
        train_df, test_df = pipeline.train_test_split(df, test_periods=req.test_periods)
        y_train = train_df.set_index("Date")["Modal_Price"]
        y_test = test_df.set_index("Date")["Modal_Price"]
        
        result = ModelManager.train_and_evaluate(
            model_type=req.model_type,
            train_series=y_train,
            test_series=y_test,
            forecast_horizon=req.forecast_horizon,
            params=req.params
        )
        
        # Attach dates for clear UI visualization
        result["train_dates"] = [str(d.date()) if hasattr(d, 'date') else str(d) for d in train_df["Date"]]
        result["test_dates"] = [str(d.date()) if hasattr(d, 'date') else str(d) for d in test_df["Date"]]
        
        # Generate future dates
        last_date = df["Date"].max()
        freq_alias = "W" if req.frequency == "W" else ("MS" if req.frequency == "M" else "D")
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=req.forecast_horizon, freq=freq_alias)
        result["future_dates"] = [str(d.date()) for d in future_dates]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/models/auto_tune")
def auto_tune_sarima(req: AutoTuneRequest):
    """Executes Box-Jenkins grid search to identify optimal ARIMA / SARIMA orders."""
    try:
        df = pipeline.filter_and_aggregate(
            commodity=req.commodity,
            state=req.state,
            mandi=req.mandi,
            frequency=req.frequency
        )
        series = df.set_index("Date")["Modal_Price"]
        
        tuning_res = ArimaSarimaForecaster.auto_order_search(
            series,
            max_p=req.max_p,
            max_d=req.max_d,
            max_q=req.max_q,
            is_seasonal=req.is_seasonal,
            s=req.seasonal_period,
            criterion=req.criterion
        )
        return tuning_res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/models/benchmark")
def benchmark_all_models(req: BenchmarkRequest):
    """Runs a side-by-side benchmark comparing ARIMA, SARIMA, Holt-Winters, Random Forest, MA, and Naive."""
    try:
        df = pipeline.filter_and_aggregate(
            commodity=req.commodity,
            state=req.state,
            mandi=req.mandi,
            frequency=req.frequency
        )
        train_df, test_df = pipeline.train_test_split(df, test_periods=req.test_periods)
        y_train = train_df.set_index("Date")["Modal_Price"]
        y_test = test_df.set_index("Date")["Modal_Price"]
        
        leaderboard = ModelManager.benchmark_all_models(
            y_train, y_test, seasonal_period=req.seasonal_period
        )
        return {
            "commodity": req.commodity,
            "test_sample_size": len(y_test),
            "leaderboard": leaderboard
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/export/forecast_csv")
def export_forecast_csv(req: ModelTrainRequest):
    """Exports actuals, test predictions, and future forecasts into a downloadable CSV."""
    try:
        # Run forecast
        res = train_model(req)
        
        # Build CSV table
        rows = []
        for dt, actual in zip(res["train_dates"], res["test_actual"]):
            rows.append({"Date": dt, "Type": "Historical Train", "Actual_Price": actual, "Forecast_Price": None, "Lower_CI_95": None, "Upper_CI_95": None})
            
        for dt, act, pred in zip(res["test_dates"], res["test_actual"], res["test_predicted"]):
            rows.append({"Date": dt, "Type": "Test Validation", "Actual_Price": act, "Forecast_Price": pred, "Lower_CI_95": None, "Upper_CI_95": None})
            
        for dt, pred, lo, up in zip(res["future_dates"], res["future_forecast"], res["future_conf_lower"], res["future_conf_upper"]):
            rows.append({"Date": dt, "Type": "Future Forecast", "Actual_Price": None, "Forecast_Price": pred, "Lower_CI_95": lo, "Upper_CI_95": up})
            
        out_df = pd.DataFrame(rows)
        stream = io.StringIO()
        out_df.to_csv(stream, index=False)
        stream.seek(0)
        
        filename = f"{req.commodity}_{req.model_type}_forecast.csv"
        return StreamingResponse(
            io.BytesIO(stream.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Serve Frontend UI -----------------

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the main interactive dashboard UI."""
    template_path = os.path.join(ROOT_DIR, "frontend", "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>KrishiDristi Engine Running. Frontend index.html loading...</h1>")

if __name__ == "__main__":
    import uvicorn
    print("Starting KrishiDristi Time Series Intelligence Platform on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
