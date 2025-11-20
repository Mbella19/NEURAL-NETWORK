# Diagnosis and Fix Report

## 1. Explainability Failure
**Issue**: The training script failed during the explainability step with the error: `Input and parameter tensors are not at the same device`.
**Cause**: The `saliency_map`, `shap_values`, and `lime_explanation` functions were receiving input tensors directly from the dataset (CPU) while the model was on the GPU (MPS/CUDA).
**Fix**: Updated `scripts/run_full_training.py` to explicitly move input tensors to the correct device (`.to(device)`) before passing them to the model.

## 2. Exploding Gradients in Phase 4 (Smart Money)
**Issue**: Phase 4 showed extremely high gradient norms (~31.8), causing instability.
**Cause**: The `CrossEntropyLoss` was using dynamically calculated class weights to handle imbalance. For extremely rare classes (like specific FVG patterns), these weights could become very large (e.g., >100x), causing gradients to explode.
**Fix**: Clamped the calculated class weights in `scripts/run_full_training.py` to a maximum of `10.0`. This preserves the rebalancing effect without introducing numerical instability.

## 3. Class Imbalance and Trivial Tasks (Phases 5, 6, 7)
**Issue**: Phases 5 (Candlesticks), 6 (Support/Resistance), and 7 (Advanced SM) showed "stuck" accuracy and extremely high positive rates (~99% predicted positives in some logs).
**Cause**: The target definition used a very low threshold (`0.0001` or 0.01%) for "significant move". In a 5-minute timeframe, price almost *always* moves more than 0.01% over 5-10 bars, making the target effectively "Always True". The model collapsed to predicting the majority class (1.0), achieving high "accuracy" but learning nothing useful.
**Fix**: Increased the thresholds in the respective phase files to `0.0015` (0.15%).
- `src/training/phases/phase5_candlesticks.py`: `profit_threshold` -> 0.0015
- `src/training/phases/phase6_sr_levels.py`: `bounce_threshold` -> 0.0015
- `src/training/phases/phase7_advanced_sm.py`: `profit_threshold` -> 0.0015

This change makes the tasks non-trivial, requiring the model to identify *truly* significant moves, which should lead to better feature learning and more meaningful predictions.

## 4. Recommendations for Future Improvements
- **Phase 1 (Direction)**: Currently binary (Up vs Not Up). Consider switching to 3-class classification (Up, Flat, Down) to separate noise from actual downward moves.
- **Phase 9 (Integration)**: The quantile loss is small and "stuck". This is expected for return forecasting. Monitor `directional_accuracy` instead of raw loss.
- **Hyperparameters**: If instability persists, try reducing the learning rate for specific phases or increasing the batch size.
