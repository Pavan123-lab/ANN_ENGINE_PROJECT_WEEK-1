import numpy as np
import pandas as pd

def generate_engine_dataset(n_samples=600, random_seed=42):
    np.random.seed(random_seed)

    inj_timing = np.random.uniform(15.0, 25.0, n_samples)       
    inj_pressure = np.random.uniform(300.0, 1000.0, n_samples)  
    afr = np.random.uniform(12.0, 24.0, n_samples)              
    comp_ratio = np.random.uniform(14.0, 18.5, n_samples)       
    egr_rate = np.random.uniform(0.0, 30.0, n_samples)          
    speed = np.random.uniform(1200.0, 3600.0, n_samples)        
    fuel_blend = np.random.uniform(0.0, 100.0, n_samples)       

    bp = (speed / 1000.0) * (inj_pressure / 150.0) * (comp_ratio / 16.0) + np.random.normal(0, 1.2, n_samples)
    bp = np.clip(bp, 5.0, 60.0)

    bte = 22.0 + (comp_ratio * 0.8) + (inj_pressure * 0.006) - (egr_rate * 0.15) - (fuel_blend * 0.03) + np.random.normal(0, 0.8, n_samples)
    bte = np.clip(bte, 20.0, 42.0)

    bsfc = (3600.0 / (bte * 0.42)) + (fuel_blend * 0.45) - (inj_pressure * 0.04) + np.random.normal(0, 8.0, n_samples)
    bsfc = np.clip(bsfc, 180.0, 450.0)

    nox = 1.2 + (inj_timing * 0.18) + (inj_pressure * 0.004) + (comp_ratio * 0.25) - (egr_rate * 0.14) + np.random.normal(0, 0.3, n_samples)
    nox = np.clip(nox, 0.8, 12.0)

    co = 0.08 + (28.0 / afr) * 0.12 + (egr_rate * 0.015) - (fuel_blend * 0.008) + np.random.normal(0, 0.04, n_samples)
    co = np.clip(co, 0.05, 4.5)

    hc = 120.0 + (32.0 / afr) * 80.0 + (egr_rate * 4.2) - (inj_pressure * 0.08) - (fuel_blend * 0.5) + np.random.normal(0, 15.0, n_samples)
    hc = np.clip(hc, 80.0, 950.0)

    smoke = 12.0 + (egr_rate * 0.75) - (afr * 0.65) - (inj_pressure * 0.012) + np.random.normal(0, 1.5, n_samples)
    smoke = np.clip(smoke, 5.0, 65.0)

    co2 = 8.5 + (bte * 0.18) - (co * 0.5) + np.random.normal(0, 0.4, n_samples)
    co2 = np.clip(co2, 6.0, 16.0)

    df = pd.DataFrame({
        # 7 Inputs
        "injection_timing": np.round(inj_timing, 2),
        "injection_pressure": np.round(inj_pressure, 1),
        "air_fuel_ratio": np.round(afr, 2),
        "compression_ratio": np.round(comp_ratio, 2),
        "egr_rate": np.round(egr_rate, 2),
        "engine_speed": np.round(speed, 0),
        "fuel_blend": np.round(fuel_blend, 1),

        "brake_power": np.round(bp, 2),
        "brake_thermal_efficiency": np.round(bte, 2),
        "bsfc": np.round(bsfc, 2),
        "nox": np.round(nox, 3),
        "co": np.round(co, 3),
        "hc": np.round(hc, 1),
        "smoke_opacity": np.round(smoke, 2),
        "co2": np.round(co2, 2)
    })

    return df

if __name__ == "__main__":
    df = generate_engine_dataset(n_samples=600)
    output_path = "data/raw/engine_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset updated successfully: {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Saved to: {output_path}")