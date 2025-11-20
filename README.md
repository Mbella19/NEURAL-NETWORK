# Advanced Self-Improving AI Day Trading System

This repository houses the implementation of a curriculum-driven hybrid
LSTM-Transformer model tailored for intraday FX trading on an M2 MacBook Air
(8GB RAM). The build follows the architecture and sequencing described in
`AGENTS.md` and `TASK.md`, emphasizing memory efficiency, logging rigor, and
progressive validation.

## Current Phase
- **Phase 1.4**: The layered configuration system (Python defaults → YAML
  overrides → `AI_TRADING_*` env vars) is active alongside the Phase 1.1-1.3
  foundations. Upcoming work will focus on data ingestion and feature
  engineering modules that consume these settings.

## Directory Layout
```
ai-trading-bot/
├── config/            # Settings module and YAML overrides
├── data/              # Raw/processed/features/checkpoints
├── logs/              # Segmented logging outputs
├── notebooks/         # Exploratory research and experiments
├── src/               # Source code packages
├── tests/             # Pytest suite (to be populated)
├── requirements.txt   # Dependency lock-step per TASK.md
└── setup.py           # Editable install script
```

## Setup Instructions
1. Ensure Python 3.10+ is installed on the M2 machine.
2. A ready-to-use virtual environment already lives at `./venv` (built with
   `pyenv`'s Python 3.10.14). Activate it via `source venv/bin/activate`.
3. Install dependencies with `pip install -r requirements.txt`. Notes:
   - `ta-lib` expects the native TA-Lib C library (`brew install ta-lib` on macOS)
     to succeed.
   - `pandas-ta==0.3.14b0` has been removed from PyPI; we will vendor an
     alternate distribution or upgrade once a stable mirror is identified.
   - `MetaTrader5` wheels are Windows-only; install them directly on the MT5
     bridge machine that will host live trading connectivity.
4. Verify TensorFlow and PyTorch detect the M2 GPU/Metal backend (see commands
   below) before continuing with subsequent phases.

### Environment Status
- `venv` contains TensorFlow 2.13.0 + `tensorflow-metal` and PyTorch 2.0.1 with
  all other Python dependencies pinned per `requirements.txt`.
- GPU checks executed via  
  `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` and  
  `python -c "import torch; print(torch.backends.mps.is_available())"`.  
  The current CLI session reports no GPU access; run the same commands from a
  local terminal (outside of Codex) to confirm Metal availability.
- `ta-lib`, `pandas-ta`, and `MetaTrader5` items require the manual steps noted
  above before `pip install -r requirements.txt` will fully succeed.

## Configuration
- `config/settings.py` defines strongly typed dataclasses for data paths, model
  hyperparameters, trading parameters, feature-engineering toggles, and logging
  policies. It auto-loads `.env` values and caches the resolved settings object.
- `config/trading_config.yaml` is the editable layer for day-to-day tweaks. Add
  new keys under the existing sections and reload via
  `python -c "from config import get_settings; print(get_settings(reload=True))"`.
- Prefix environment variables with `AI_TRADING_` (e.g. `AI_TRADING_ENV=prod`,
  `AI_TRADING_LOG_LEVEL=DEBUG`, `AI_TRADING_TRAINING_DATA=/path/to/data`) to
  override YAML/Python defaults without touching source control.

## Data Sources
Historical training data is stored externally at:
```
/Users/gervaciusjr/Desktop/AI Trading Bot/Training data
```
The `config/settings.py` module references this path so loaders can stream the
data without duplicating large files inside the repository.

### Loading EURUSD Data (Phase 2.2)
Use the new ingestion helper to validate and stage the canonical EURUSD file:
```python
from data_loader import ingest_training_data

result = ingest_training_data(
    file_name="EURUSD_M1_202306010000_202412302358.csv",
    copy_to_raw=True,
)
print(result.metadata)
```
This will:
1. Load and validate the TSV/CSV file from the external training directory.
2. Copy it into `data/raw/` for versioned processing.
3. Write a JSON data-quality report under `data/raw/reports/`.

### Timeframe Aggregation (Phase 2.3)
After ingesting, build the higher-timeframe datasets in `data/processed/timeframes/`:
```python
from data_loader.timeframe_aggregator import aggregate_timeframes_from_raw

aggregate_timeframes_from_raw(timeframes=("M5", "M15", "H1"))
```
This script:
1. Loads the staged M1 data from `data/raw/`.
2. Aggregates OHLCV bars into M5/M15/H1 sequences (first open, last close, max
   high, min low, summed volume/tick volume, averaged spread).
3. Emits CSV files (falls back to CSV if Parquet engines are unavailable) plus
   warnings whenever large timestamp gaps exist (e.g., weekends or market
   holidays).

### Preprocessing Pipeline (Phase 2.4)
Normalize and split the cleaned dataset for downstream model training:
```python
from data_loader.preprocessor import preprocess_dataset

splits = preprocess_dataset(
    outlier_method="zscore",
    normalization="zscore",
    split_ratios=(0.7, 0.15, 0.15),
)
print(len(splits.train), len(splits.val), len(splits.test))
```
Outputs land in `data/processed/preprocessed/`:
```
train.csv / val.csv / test.csv
scalers.json  # training-set statistics for consistent normalization
```
The pipeline removes weekends/holidays, fills missing numeric values via
forward/backward fill, clamps outliers, and ensures splits are purely
time-ordered to avoid leakage.

### Data Quality Assurance (Phase 2.5)
Validate the staged dataset before progressing to feature engineering:
```python
from data_loader.quality_checker import run_quality_checks

report = run_quality_checks(price_gap_threshold=0.0005)
print(report.summary)
```
This writes `data/raw/reports/data_quality_summary.json` summarizing:
- look-ahead bias checks (timestamp monotonicity, duplicates)
- price continuity gaps between consecutive candles
- volume sanity (negative/zero ratios)
- anomaly detection (extreme z-score outliers)

## Feature Engineering (Phase 3)
- Core calculators live in `src/feature_engineering/` and all inherit from
  `BaseFeatureCalculator` for consistent validation.
- Highlighted modules:
  - `time_features.py`: cyclical hour/day encodings, session flags, holiday
    markers.
  - `technical_indicators.py`: SMA/EMA, RSI, MACD, stochastic, ATR/ADX, volume
    smoothing, efficiency ratio.
  - `wick_features.py`, `candlestick_patterns.py`, `market_structure.py`,
    `smart_money.py`, `support_resistance.py`, `market_regime.py`.
- Use the default pipeline to compute every feature in one pass:
  ```python
  from feature_engineering.pipeline import build_default_feature_pipeline, run_feature_pipeline

  pipeline = build_default_feature_pipeline()
  result = run_feature_pipeline(df, pipeline=pipeline)
  feature_df = result.dataframe
  ```
- For large datasets, stream batches via `run_pipeline_on_iterator()` to keep
  memory usage under control.
- Explore `notebooks/feature_diagnostics.ipynb` to compare feature stability
  across (M1/M5/M15/H1) datasets and visualize session overlaps vs volatility
  regimes before feeding signals into curriculum phases.

## Next Steps
- Connect the configuration system to Phase 2 data loaders and feature modules.
- Vendor/replace the missing `pandas-ta==0.3.14b0` wheel and install TA-Lib's C
  library so the requirements file can succeed end-to-end.
- Kick off Phase 2 (MT5 data extraction + preprocessing) now that setup,
  dependencies, and configuration primitives are complete.
- Begin Phase 3 feature engineering by subclassing `BaseFeatureCalculator` in
  `src/feature_engineering/`. Register calculators with `FeaturePipeline` to
  produce modular, testable, memory-aware features aligned with AGENTS.md.
  - Example: `WickFeatures` (Phase 3.2) calculates wick/body ratios and
    dominance metrics while respecting the base-class validation hooks.
