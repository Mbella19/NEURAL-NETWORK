import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from training.multitask_dataset import MultiTaskDataset

def test_normalization():
    print("Testing normalization logic...")
    
    # Create dummy data
    data = {
        "TIMESTAMP": pd.date_range(start="2023-01-01", periods=100, freq="5T"),
        "OPEN": np.linspace(100, 110, 100),
        "HIGH": np.linspace(101, 111, 100),
        "LOW": np.linspace(99, 109, 100),
        "CLOSE": np.linspace(100.5, 110.5, 100),
        "SMA_14": np.linspace(100, 110, 100) * 1.01, # 1% above close
        "EMA_14": np.linspace(100, 110, 100) * 0.99, # 1% below close
        "RSI": np.random.rand(100) * 100
    }
    df = pd.DataFrame(data)
    
    # Apply normalization
    normalized = MultiTaskDataset._normalize_price_features(df)
    
    # Check CLOSE
    close_vals = normalized["CLOSE"].values
    print(f"CLOSE values (first 5): {close_vals[:5]}")
    if np.all(close_vals == 0.0):
        print("FAIL: CLOSE is all 0.0!")
    else:
        print("PASS: CLOSE is not all 0.0 (contains returns)")
        
    # Check SMA
    sma_vals = normalized["SMA_14"].values
    print(f"SMA values (first 5): {sma_vals[:5]}")
    # Expected: (SMA - CLOSE) / CLOSE ~ (1.01*C - C)/C = 0.01
    if np.allclose(sma_vals, 0.01, atol=1e-3):
        print("PASS: SMA is normalized relative to CLOSE")
    else:
        print(f"FAIL: SMA values unexpected. Mean: {sma_vals.mean()}")

    # Check EMA
    ema_vals = normalized["EMA_14"].values
    print(f"EMA values (first 5): {ema_vals[:5]}")
    # Expected: (EMA - CLOSE) / CLOSE ~ (0.99*C - C)/C = -0.01
    if np.allclose(ema_vals, -0.01, atol=1e-3):
        print("PASS: EMA is normalized relative to CLOSE")
    else:
        print(f"FAIL: EMA values unexpected. Mean: {ema_vals.mean()}")

if __name__ == "__main__":
    test_normalization()
