import sys
import os
sys.path.append(os.path.abspath("."))
from src.data_pipeline import CropDataPipeline
from src.models.model_manager import ModelManager

p = CropDataPipeline()
df = p.filter_and_aggregate("Wheat", frequency="W")
print("Filtered shape:", df.shape)

tr, te = p.train_test_split(df, test_periods=20)
y_tr = tr.set_index("Date")["Modal_Price"]
y_te = te.set_index("Date")["Modal_Price"]

print("Training models...")
res = ModelManager.train_and_evaluate("arima", y_tr, y_te, forecast_horizon=8, params={"p": 1, "d": 1, "q": 1})
print("ARIMA metrics:", res["metrics"])

bench = ModelManager.benchmark_all_models(y_tr, y_te, seasonal_period=12)
print(f"Benchmark finished with {len(bench)} models:")
for b in bench:
    print(f"Rank {b['rank']}: {b['model_name']} -> RMSE={b['rmse']}, MAPE={b['mape']}%, Acc={b['accuracy_pct']}%")
