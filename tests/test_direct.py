"""
Direct Endpoint Function Verification Script
Calls FastAPI route functions directly without relying on Starlette TestClient.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import (
    get_dataset_metadata,
    get_dataset_health,
    clean_selected_series,
    get_time_series,
    analyze_stationarity,
    decompose_time_series,
    train_model,
    benchmark_all_models,
    serve_index,
    login_with_gmail,
    get_current_session,
    logout_user,
    GmailLoginRequest,
    FilterRequest,
    CleanRequest,
    StationarityRequest,
    DecompositionRequest,
    ModelTrainRequest,
    BenchmarkRequest
)

def run_direct_tests():
    print("[1/9] Testing serve_index() (KrishiDristi HTML UI template)...")
    html_res = serve_index()
    assert "KrishiDristi" in html_res.body.decode("utf-8")
    print(f"      ✓ Index HTML served successfully ({len(html_res.body)} bytes).")

    print("[2/9] Testing Gmail Authentication (login_with_gmail)...")
    auth_res = login_with_gmail(GmailLoginRequest(email="aayush.test@gmail.com", name="Aayush Test"))
    assert auth_res["success"] is True
    assert auth_res["user"]["email"] == "aayush.test@gmail.com"
    token = auth_res["user"]["token"]
    # Check session
    sess = get_current_session(token)
    assert sess["authenticated"] is True
    assert sess["user"]["name"] == "Aayush Test"
    print(f"      ✓ Gmail Login Success: {sess['user']['name']} ({sess['user']['email']}) | Token: {token[:12]}...")

    print("[3/9] Testing get_dataset_metadata()...")
    meta = get_dataset_metadata()
    assert meta["total_records"] >= 100000
    assert "Wheat" in meta["commodities"]
    assert "Tomato" in meta["commodities"]
    print(f"      ✓ Verified 100K+ Dataset: {meta['total_records']:,} observations across {len(meta['commodities'])} crops.")

    print("[4/9] Testing Data Cleaning & Health Audit (get_dataset_health & clean_selected_series)...")
    health = get_dataset_health()
    assert health["health_score"] >= 90.0
    print(f"      ✓ Dataset Health Score: {health['health_score']}/100 | Outliers: {health['outliers_identified']} ({health['outlier_pct']}%)")
    clean_res = clean_selected_series(CleanRequest(commodity="Wheat", outlier_method="iqr", treatment="winsorize"))
    assert len(clean_res["cleaned_series"]) == len(clean_res["raw_series"])
    print(f"      ✓ Outlier Treatment: {clean_res['treatment']} | Variance Reduced by {clean_res['stats_comparison']['variance_reduction_pct']}%")

    print("[5/9] Testing get_time_series() (Weekly Filter)...")
    ts = get_time_series(FilterRequest(commodity="Wheat", frequency="W"))
    assert ts["count"] > 100
    assert len(ts["dates"]) == len(ts["modal_prices"])
    print(f"      ✓ Aggregated Series: {ts['count']} steps ({ts['dates'][0]} to {ts['dates'][-1]}).")

    print("[6/9] Testing analyze_stationarity() with Differencing d=1...")
    st = analyze_stationarity(StationarityRequest(commodity="Wheat", diff_order=1, frequency="W"))
    assert st["tests"]["adf"]["is_stationary"] is True
    print(f"      ✓ ADF Stat: {st['tests']['adf']['test_statistic']} (p={st['tests']['adf']['p_value']}) -> {st['tests']['joint_verdict']}")

    print("[7/9] Testing decompose_time_series() (STL Decomposition)...")
    dec = decompose_time_series(DecompositionRequest(commodity="Wheat", method="stl", period=52))
    assert dec["metrics"]["trend_strength"] > 0.8
    print(f"      ✓ STL Decomposed: Trend Strength = {dec['metrics']['trend_strength']}, Seasonal Strength = {dec['metrics']['seasonality_strength']}")

    print("[8/9] Testing train_model() (ARIMA Model)...")
    train_res = train_model(ModelTrainRequest(
        commodity="Wheat", frequency="W", model_type="arima",
        test_periods=15, forecast_horizon=8, params={"p": 1, "d": 1, "q": 1}
    ))
    assert train_res["metrics"]["accuracy_score_pct"] > 85.0
    assert len(train_res["future_forecast"]) == 8
    print(f"      ✓ Model: {train_res['model_name']} | Accuracy: {train_res['metrics']['accuracy_score_pct']}% | Future Horizon: {len(train_res['future_forecast'])} weeks")

    print("[9/9] Testing benchmark_all_models()...")
    bench_res = benchmark_all_models(BenchmarkRequest(commodity="Wheat", frequency="W", test_periods=12, seasonal_period=12))
    assert len(bench_res["leaderboard"]) >= 4
    top = bench_res["leaderboard"][0]
    print(f"      ✓ Leaderboard Top Rank: {top['model_name']} with RMSE=₹{top['rmse']} and Accuracy={top['accuracy_pct']}%")

    print("\n" + "=" * 60)
    print("  🏆 ALL KRISHIDRISTI AUTH & TIME SERIES PIPELINES VERIFIED!")
    print("=" * 60)

if __name__ == "__main__":
    run_direct_tests()
