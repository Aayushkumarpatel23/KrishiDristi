"""
End-to-End Verification Test Script
Tests FastAPI endpoints directly with verbose progress reporting.
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app import app

def run_tests():
    print("[1/6] Initializing TestClient...")
    client = TestClient(app)
    
    print("[2/6] Testing GET / (HTML Dashboard)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "AgriForecast" in res.text
    print("      ✓ UI Dashboard served successfully (Length: %d bytes)" % len(res.text))

    print("[3/6] Testing GET /api/dataset/metadata...")
    res = client.get("/api/dataset/metadata")
    assert res.status_code == 200
    meta = res.json()
    assert meta["total_records"] >= 100000
    print(f"      ✓ Total Records: {meta['total_records']:,} | Crops: {len(meta['commodities'])}")

    print("[4/6] Testing POST /api/dataset/timeseries (Weekly Aggregation)...")
    res = client.post("/api/dataset/timeseries", json={"commodity": "Wheat", "frequency": "W"})
    assert res.status_code == 200
    ts = res.json()
    print(f"      ✓ Aggregated Time Steps: {ts['count']} | Dates: {ts['dates'][0]} to {ts['dates'][-1]}")

    print("[5/6] Testing POST /api/analysis/stationarity (ADF & KPSS)...")
    res = client.post("/api/analysis/stationarity", json={"commodity": "Wheat", "diff_order": 1, "frequency": "W"})
    assert res.status_code == 200
    stat = res.json()
    print(f"      ✓ ADF Stat: {stat['tests']['adf']['test_statistic']} (p={stat['tests']['adf']['p_value']})")
    print(f"      ✓ Joint Verdict: {stat['tests']['joint_verdict']}")

    print("[6/6] Testing POST /api/models/train (ARIMA Model)...")
    res = client.post("/api/models/train", json={
        "commodity": "Wheat", "frequency": "W",
        "model_type": "arima", "test_periods": 15, "forecast_horizon": 8,
        "params": {"p": 1, "d": 1, "q": 1}
    })
    assert res.status_code == 200
    model = res.json()
    print(f"      ✓ Model: {model['model_name']}")
    print(f"      ✓ Accuracy: {model['metrics']['accuracy_score_pct']}% (RMSE: ₹{model['metrics']['rmse']})")
    print(f"      ✓ Future Forecast Horizons: {len(model['future_forecast'])} weeks")

    print("\n🎉 ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
