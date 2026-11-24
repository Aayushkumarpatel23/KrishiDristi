"""
KrishiDristi Startup Runner
Checks dataset status and initializes the FastAPI server on http://127.0.0.1:8000
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.generate_dataset import generate_crop_dataset

def main():
    print("=" * 70)
    print("  🌾 KRISHIDRISTI: CROP PRICE TIME SERIES INTELLIGENCE SYSTEM")
    print("=" * 70)
    
    data_dir = os.path.join(BASE_DIR, "data")
    parquet_path = os.path.join(data_dir, "crop_prices_100k.parquet")
    
    if not os.path.exists(parquet_path):
        print("\n[+] Large dataset (100K+ entries) not detected. Generating now...")
        generate_crop_dataset(data_dir)
        print("[+] Dataset generated successfully.\n")
    else:
        print(f"[✓] Verified existing 100K+ dataset at: {parquet_path}")

    print("\n[+] Launching FastAPI Web Dashboard & REST Analytics Server...")
    print("    URL: http://127.0.0.1:8000")
    print("    API Docs: http://127.0.0.1:8000/docs")
    print("=" * 70)
    print("Press Ctrl+C to stop the server at any time.\n")
    
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
