import sys
from pathlib import Path
import ast

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

def check_file_content(file_path, checks):
    print(f"\nChecking {file_path.name}...")
    try:
        content = file_path.read_text()
    except FileNotFoundError:
        print(f"  ✗ FAILED: File not found: {file_path}")
        return False

    all_passed = True
    for check_name, check_fn in checks.items():
        if check_fn(content):
            print(f"  ✓ PASS: {check_name}")
        else:
            print(f"  ✗ FAIL: {check_name}")
            all_passed = False
    return all_passed

def verify_multitask_dataset():
    file_path = REPO_ROOT / "src/training/multitask_dataset.py"
    
    def check_zeroed_close_fix(content):
        return "df[col] = df[col].pct_change().fillna(0.0)" in content

    def check_normalization_fix(content):
        return 'price_markers = ("OPEN", "HIGH", "LOW", "CLOSE", "SMA", "EMA")' in content

    return check_file_content(file_path, {
        "Zeroed Close Fix": check_zeroed_close_fix,
        "Indicator Normalization Fix": check_normalization_fix
    })

def verify_phase1_direction():
    file_path = REPO_ROOT / "src/training/phases/phase1_direction.py"
    
    def check_horizon_fix(content):
        return "horizon = max(1, self.config.forecast_horizon)" in content

    return check_file_content(file_path, {
        "Horizon Mismatch Fix": check_horizon_fix
    })

def verify_multitask_model():
    file_path = REPO_ROOT / "src/models/multitask.py"
    
    def check_mlp_head(content):
        return "nn.Linear(feat_dim, feat_dim)" in content and "nn.GELU()" in content

    return check_file_content(file_path, {
        "2-Layer MLP Head Upgrade": check_mlp_head
    })

def verify_run_full_training():
    file_path = REPO_ROOT / "scripts/run_full_training.py"
    
    def check_zeroed_close_fix(content):
        return "df[col] = df[col].pct_change().fillna(0.0)" in content

    def check_normalization_fix(content):
        return 'price_markers = ("OPEN", "HIGH", "LOW", "CLOSE", "SMA", "EMA")' in content

    def check_broken_tasks_disabled(content):
        # Check if Phase 5, 6, 7 are commented out in all_tasks list
        # This is a bit heuristic, looking for the commented out lines
        return (
            "# Phase5CandlestickTask()" in content and
            "# Phase6SupportResistanceTask()" in content and
            "# Phase7AdvancedSMTask()" in content
        )

    def check_static_weighting(content):
        return 'WEIGHTING_STRATEGY = "static"' in content

    def check_learning_rate(content):
        return 'mt_learning_rate = 1e-4' in content

    def check_warmup(content):
        return 'ENABLE_WARMUP = True' in content

    def check_fsattention_disabled(content):
        return 'use_fsatten=False' in content
    
    def check_task_weights(content):
        return '"Phase1DirectionTask": 3.0' in content and '"Phase2IndicatorTask": 2.0' in content

    return check_file_content(file_path, {
        "Zeroed Close Fix": check_zeroed_close_fix,
        "Indicator Normalization Fix": check_normalization_fix,
        "Broken Tasks Disabled": check_broken_tasks_disabled,
        "Static Weighting": check_static_weighting,
        "Learning Rate Reduced": check_learning_rate,
        "Warmup Enabled": check_warmup,
        "FSAttention Disabled": check_fsattention_disabled,
        "Task Weights Updated": check_task_weights
    })

def main():
    print("========================================================")
    print("VERIFYING ALL FIXES")
    print("========================================================")
    
    dataset_passed = verify_multitask_dataset()
    phase1_passed = verify_phase1_direction()
    model_passed = verify_multitask_model()
    training_passed = verify_run_full_training()
    
    print("\n========================================================")
    if dataset_passed and phase1_passed and model_passed and training_passed:
        print("✓ ALL CHECKS PASSED: Codebase is correctly patched.")
    else:
        print("✗ SOME CHECKS FAILED: Please review the failures above.")
    print("========================================================")

if __name__ == "__main__":
    main()
