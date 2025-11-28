# Training Diagnosis & Fix Report

Based on the analysis of your training logs and codebase, here are the key findings and recommendations.

## 1. The "Stuck" Phase 1 (Direction Task)
**Status:** The Direction task is stuck at a loss of ~0.693, which corresponds to random guessing (predicting 0.5 probability for everything).
**Cause:** The target generation logic falls back to "Terminal Price Direction" when neither Take Profit (TP) nor Stop Loss (SL) is hit within 50 bars.
```python
if tp_first is None and sl_first is None:
    # Neither hit: fall back to direction of terminal price in window
    terminal = close.iloc[end - 1]
    labels[idx] = 1.0 if terminal > close.iloc[idx] else 0.0
```
In ranging markets, this fallback logic introduces significant noise (labeling random fluctuations as "trends"). The model correctly learns to ignore this noise by outputting 0.5.
**Recommendation:**
- **Increase `time_limit`**: Increase `time_limit` in `Phase1Config` (e.g., to 100 or 200 bars) to allow more trades to resolve naturally.
- **Add "Neutral" Class**: Instead of forcing a 0/1 label on unresolved trades, mark them as ignored (mask them out) or introduce a 3rd "Neutral" class.

## 2. The "Gradient Bully" Phase 4 (Smart Money)
**Status:** Phase 4 (SMC) has extremely high gradient norms (up to 3.0) compared to other tasks (0.1 - 0.2), despite having a lower task weight (0.1).
**Cause:** It uses `CrossEntropyLoss` on 3 classes, which generates larger gradients than the `BCEWithLogitsLoss` used by other tasks. This "loud" gradient is drowning out the signal from subtle tasks like Direction.
**Recommendation:**
- **Reduce Weight Further**: Lower `Phase4SmartMoneyTask` weight in `task_weights` to `0.01` or `0.05`.
- **Gradient Clipping**: The `MultiTaskTrainingLoop` has `normalize_gradients=True`, but the `gradient_norm_target=1.0` is likely too high for the other tasks to compete with Phase 4. Try setting `gradient_norm_target=0.5`.

## 3. Phase 3 (Structure) is Promising
**Status:** Loss is ~1.39. Random guessing for 5 classes (HH, HL, LH, LL, NONE) is ln(5) ≈ 1.61.
**Finding:** The model *is* learning market structure! This is a strong foundation.

## 4. Missing `models` Directory
**Observation:** The `src/models` directory was not visible in the file list, yet the code runs. This suggests it exists in your environment but might be ignored by git or located elsewhere in `PYTHONPATH`. Ensure `models/temporal_fusion.py` and `models/multitask.py` are committed if you plan to share the repo.

## 5. Action Plan
1.  **Modify `Phase1DirectionTask`**: Increase `time_limit` to `100` in `src/training/phases/phase1_direction.py`.
2.  **Modify `run_full_training.py`**:
    - Change `Phase4SmartMoneyTask` weight to `0.05`.
    - Change `gradient_norm_target` to `0.5` in `mt_config`.
3.  **Run Sanity Check**: If possible, try training *only* Phase 1 (disable others in `all_tasks` list) to see if it can overfit a small batch. This confirms the model architecture is capable of learning the task.
