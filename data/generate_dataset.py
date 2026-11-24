"""
High-Volume Realistic Agricultural Crop Price Dataset Generator
Generates 120,000+ historical mandi transaction records (2018-2024)
incorporating realistic economic trends, multi-harmonic seasonality,
arrival-supply dynamics, meteorological shocks, and autoregressive memory.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_crop_dataset(output_dir: str, target_records: int = 120000, seed: int = 42):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Commodities and their specific economic profiles
    commodities_spec = {
        "Wheat": {
            "base_price": 2050,
            "trend_rate": 0.052, # ~5.2% annual growth
            "volatility": 0.06,
            "harvest_month": 4, # April (Rabi)
            "seasonal_amplitude": 0.12, # 12% price swing across seasons
            "states": ["Punjab", "Haryana", "Madhya Pradesh", "Uttar Pradesh"],
            "mandis": {
                "Punjab": ["Khanna", "Ludhiana"],
                "Haryana": ["Karnal", "Ambala"],
                "Madhya Pradesh": ["Indore", "Ujjain"],
                "Uttar Pradesh": ["Kanpur", "Agra"]
            }
        },
        "Paddy (Rice)": {
            "base_price": 1950,
            "trend_rate": 0.048,
            "volatility": 0.07,
            "harvest_month": 11, # November (Kharif)
            "seasonal_amplitude": 0.14,
            "states": ["Punjab", "Haryana", "Andhra Pradesh", "Uttar Pradesh"],
            "mandis": {
                "Punjab": ["Jalandhar", "Patiala"],
                "Haryana": ["Kurukshetra", "Sirsa"],
                "Andhra Pradesh": ["Vijayawada", "Guntur"],
                "Uttar Pradesh": ["Bareilly", "Varanasi"]
            }
        },
        "Maize": {
            "base_price": 1650,
            "trend_rate": 0.055,
            "volatility": 0.09,
            "harvest_month": 10,
            "seasonal_amplitude": 0.16,
            "states": ["Madhya Pradesh", "Maharashtra", "Karnataka"],
            "mandis": {
                "Madhya Pradesh": ["Chhindwara", "Dewas"],
                "Maharashtra": ["Sangli", "Kolhapur"],
                "Karnataka": ["Davangere", "Shimoga"]
            }
        },
        "Cotton": {
            "base_price": 5800,
            "trend_rate": 0.065,
            "volatility": 0.12,
            "harvest_month": 11,
            "seasonal_amplitude": 0.18,
            "states": ["Maharashtra", "Gujarat", "Telangana"],
            "mandis": {
                "Maharashtra": ["Nagpur", "Amravati"],
                "Gujarat": ["Rajkot", "Surendranagar"],
                "Telangana": ["Warangal", "Adilabad"]
            }
        },
        "Soybean": {
            "base_price": 4200,
            "trend_rate": 0.060,
            "volatility": 0.11,
            "harvest_month": 10,
            "seasonal_amplitude": 0.17,
            "states": ["Madhya Pradesh", "Maharashtra", "Rajasthan"],
            "mandis": {
                "Madhya Pradesh": ["Indore", "Sehore"],
                "Maharashtra": ["Latur", "Akola"],
                "Rajasthan": ["Kota", "Baran"]
            }
        },
        "Onion": {
            "base_price": 1900,
            "trend_rate": 0.075,
            "volatility": 0.28, # High volatility commodity
            "harvest_month": 3, # March Rabi, second flush Oct-Nov
            "seasonal_amplitude": 0.35, # Large price swings
            "states": ["Maharashtra", "Madhya Pradesh", "Karnataka"],
            "mandis": {
                "Maharashtra": ["Lasalgaon", "Nashik", "Pune"],
                "Madhya Pradesh": ["Neemuch", "Khandwa"],
                "Karnataka": ["Hubli", "Belgaum"]
            }
        },
        "Potato": {
            "base_price": 1250,
            "trend_rate": 0.045,
            "volatility": 0.22,
            "harvest_month": 2, # Feb-March
            "seasonal_amplitude": 0.28,
            "states": ["Uttar Pradesh", "West Bengal", "Punjab"],
            "mandis": {
                "Uttar Pradesh": ["Agra", "Farrukhabad"],
                "West Bengal": ["Hooghly", "Burdwan"],
                "Punjab": ["Jalandhar", "Hoshiarpur"]
            }
        },
        "Tomato": {
            "base_price": 2100,
            "trend_rate": 0.080,
            "volatility": 0.36, # Perishable, frequent price spikes
            "harvest_month": 1, # Multiple mini harvests
            "seasonal_amplitude": 0.40,
            "states": ["Karnataka", "Maharashtra", "Andhra Pradesh"],
            "mandis": {
                "Karnataka": ["Kolar", "Bangalore"],
                "Maharashtra": ["Narayangaon", "Sangamner"],
                "Andhra Pradesh": ["Madanapalle", "Anantapur"]
            }
        }
    }

    # 2. Date Range: 7 years of trading days (excluding Sundays: ~312 trading days/yr)
    start_date = pd.Timestamp("2018-01-01")
    end_date = pd.Timestamp("2024-12-31")
    full_date_range = pd.date_range(start_date, end_date, freq="D")
    # Keep Monday-Saturday (trading days in Indian APMC Mandis)
    trading_dates = [d for d in full_date_range if d.weekday() != 6]
    n_days = len(trading_dates)
    day_indices = np.arange(n_days)

    print(f"Generating time series across {n_days} trading days from {start_date.date()} to {end_date.date()}...")

    records = []
    
    # Exogenous systemic time series: Fuel / Transportation Index & Macro Inflation
    annual_trend_base = (day_indices / 365.25)
    diesel_index = 70.0 + 3.8 * annual_trend_base + 4.0 * np.sin(2 * np.pi * day_indices / 365.25) + np.random.normal(0, 1.5, n_days)
    
    # 3. Simulate for each commodity and its mandis
    for commodity_name, spec in commodities_spec.items():
        base_p = spec["base_price"]
        trend_r = spec["trend_rate"]
        vol = spec["volatility"]
        harvest_m = spec["harvest_month"]
        amp = spec["seasonal_amplitude"]
        
        # Secular growth
        trend_component = base_p * np.exp(trend_r * annual_trend_base)
        
        # Seasonal component: Harmonic representation
        # Peak supply = harvest month -> lowest price. Peak scarcity = 6 months later -> peak price.
        day_of_year = np.array([getattr(d, 'day_of_year', d.dayofyear) for d in trading_dates])
        harvest_peak_doy = harvest_m * 30.5
        # Phase shift: lowest price around harvest_peak_doy
        seasonal_phase = 2 * np.pi * (day_of_year - harvest_peak_doy) / 365.25
        seasonality = - amp * np.cos(seasonal_phase) + 0.3 * amp * np.sin(2 * seasonal_phase)
        
        # Iterate over all states and mandis
        for state, mandis in spec["mandis"].items():
            state_multiplier = np.random.uniform(0.96, 1.05)
            
            for mandi in mandis:
                mandi_multiplier = np.random.uniform(0.97, 1.03)
                
                # Autoregressive process for daily price innovation: AR(1) with rho = 0.88
                rho = 0.88
                ar_noise = np.zeros(n_days)
                for t in range(1, n_days):
                    ar_noise[t] = rho * ar_noise[t-1] + np.random.normal(0, vol * base_p * np.sqrt(1 - rho**2))
                
                # Weather / unseasonal rain shocks (especially on Tomato/Onion)
                shock_events = np.random.binomial(1, 0.015, n_days) # 1.5% probability of weather shock
                shocks = shock_events * np.random.uniform(0.08, 0.28, n_days) * base_p
                
                # Combine: Modal Price
                raw_modal = (trend_component * (1.0 + seasonality) * state_multiplier * mandi_multiplier) + ar_noise + shocks
                # Add minor diesel transport influence
                raw_modal += (diesel_index - 70.0) * (base_p / 1000.0) * 1.5
                raw_modal = np.maximum(raw_modal, base_p * 0.45) # Floor at 45% of base
                modal_price = np.round(raw_modal, 2)
                
                # Arrival Volume (Metric Tonnes): Inversely correlated with price spikes, peaks at harvest
                base_arrival = np.random.uniform(150, 600)
                harvest_boost = (np.cos(seasonal_phase) + 1.0) * 0.5 * base_arrival * 1.8
                daily_arrival = np.maximum(20.0, base_arrival + harvest_boost - (modal_price - base_p)*0.08 + np.random.normal(0, 35, n_days))
                daily_arrival = np.round(daily_arrival, 1)
                
                # Min and Max Prices based on daily intra-market spread
                spread = np.random.uniform(0.04, 0.12, n_days) * modal_price
                min_price = np.round(modal_price - spread * np.random.uniform(0.4, 0.7, n_days), 2)
                max_price = np.round(modal_price + spread * np.random.uniform(0.5, 0.9, n_days), 2)
                
                # Rainfall deviation (%)
                monsoon_month_mask = np.isin([d.month for d in trading_dates], [6, 7, 8, 9])
                rainfall_anomaly = np.where(monsoon_month_mask, np.random.normal(0, 18, n_days), np.random.normal(0, 5, n_days))
                rainfall_anomaly = np.round(rainfall_anomaly, 1)
                
                for i, dt in enumerate(trading_dates):
                    records.append({
                        "Date": dt,
                        "Year": dt.year,
                        "Month": dt.month,
                        "Day": dt.day,
                        "DayOfWeek": dt.strftime("%A"),
                        "Commodity": commodity_name,
                        "State": state,
                        "Mandi": mandi,
                        "Arrivals_Tons": daily_arrival[i],
                        "Min_Price": min_price[i],
                        "Max_Price": max_price[i],
                        "Modal_Price": modal_price[i],
                        "Rainfall_Anomaly_Pct": rainfall_anomaly[i],
                        "Transport_Fuel_Index": round(diesel_index[i], 2)
                    })
                    
    df = pd.DataFrame(records)
    total_rows = len(df)
    print(f"Generated {total_rows:,} records.")
    
    # Save outputs
    parquet_path = os.path.join(output_dir, "crop_prices_100k.parquet")
    sample_csv_path = os.path.join(output_dir, "crop_prices_sample.csv")
    summary_path = os.path.join(output_dir, "dataset_metadata.json")
    
    df.to_parquet(parquet_path, index=False)
    # Save a 10,000 row CSV sample for quick external viewing
    df.head(10000).to_csv(sample_csv_path, index=False)
    
    # Dataset statistics
    summary = {
        "total_records": total_rows,
        "date_range": [str(start_date.date()), str(end_date.date())],
        "commodities": df["Commodity"].unique().tolist(),
        "states": df["State"].unique().tolist(),
        "mandis": df["Mandi"].unique().tolist(),
        "record_counts_by_commodity": df["Commodity"].value_counts().to_dict(),
        "average_modal_prices": df.groupby("Commodity")["Modal_Price"].mean().round(2).to_dict()
    }
    
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Saved Parquet: {parquet_path}")
    print(f"Saved CSV Sample: {sample_csv_path}")
    print(f"Saved Summary: {summary_path}")
    return total_rows

if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    generate_crop_dataset(out)
