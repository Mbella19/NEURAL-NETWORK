# Neural Network Diagnosis and Fixes

## Diagnosis

### 1. Multi-Task Interference ("Gradient Bully")
**Symptom**: Phase 4 (Smart Money) and Phase 3 (Structure) gradients are 100-1000x larger than Phase 1 (Direction) and Phase 5 (Candlesticks).
**Cause**: Static weighting combined with vastly different loss scales.
- Phase 4 (CrossEntropy) and Phase 3 (CrossEntropy) generate large losses/gradients.
- Phase 1 and 5 (BCE) generate small losses.
- The optimizer prioritizes the large gradients, effectively ignoring the core trading tasks.

### 2. Data Handling Error in Phase 3 (Structure)
**Symptom**: Phase 3 loss is extremely high (~10.0) and doesn't converge.
**Cause**: Silent data deletion.
- Structure labels ("HH", "LL", etc.) were strings.
- `pd.to_numeric(..., errors='coerce')` turned them into `NaN`, then `fillna(0)` turned them to 0.
- The model was trying to learn from zeroed-out features.

### 3. Stuck Accuracy in Phase 4 (Smart Money)
**Symptom**: Accuracy stuck at ~98.4% (majority class).
**Cause**: Class Imbalance.
- Model learned to predict "Neutral" for everything.

### 4. Overfitting in Phase 8 (Risk)
**Symptom**: Validation loss >> Training loss.
**Cause**: `MSELoss` sensitivity to outliers in the noisy `drawdown/ATR` target.

### 5. Noisy Targets in Phase 1 (Direction)
**Symptom**: Phase 1 accuracy stuck around 50%.
**Cause**: Treating micro-moves (noise) as valid "Up"/"Down" signals confuses the model.

## Implemented Solutions

### 1. Uncertainty Weighting (Fixes Interference)
- **Action**: Updated `scripts/run_full_training.py`.
- **Details**: Switched `weighting_strategy` to `"uncertainty"`.
- **Effect**: The model learns a learnable variance parameter for each task, automatically scaling down high-loss tasks (Phase 3, 4) and scaling up low-loss tasks (Phase 1, 5).

### 2. Correct Categorical Encoding (Fixes Phase 3)
- **Action**: Updated `src/training/phases/phase3_structure.py`.
- **Details**: Explicitly mapped "HH", "LL", etc., to integers before numeric conversion.
- **Effect**: The model now receives valid market structure signals.

### 3. Class Weighting (Fixes Phase 4 Imbalance)
- **Action**: Updated `scripts/run_full_training.py`.
- **Details**: Added inverse frequency weighting for `CrossEntropyLoss`.
- **Effect**: Forces the model to learn rare patterns.

### 4. Robust Loss (Fixes Phase 8 Overfitting)
- **Action**: Updated `src/training/phases/phase8_risk.py`.
- **Details**: Switched to `SmoothL1Loss`.
- **Effect**: Reduces penalty for outliers, stabilizing training.

### 5. Target Thresholding (Fixes Phase 1 Noise)
- **Action**: Updated `src/training/phases/phase1_direction.py`.
- **Details**: Added `0.0001` threshold. Only moves > 1 pip are considered "Up".
- **Effect**: Filters out noise, giving the model a cleaner signal to learn.

## Verification Steps
1. Run `scripts/run_full_training.py`.
2. Monitor "Gradient norm" in logs - they should now be comparable across tasks.
3. Monitor Phase 1 accuracy - it should break the 50% random guess barrier.
