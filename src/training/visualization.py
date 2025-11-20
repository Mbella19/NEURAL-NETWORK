"""Training visualization and analysis tools for multi-task learning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


class TrainingVisualizer:
    """Visualize multi-task training metrics and analysis.

    Features:
    - Per-task loss curves
    - Task weight evolution
    - Task health monitoring
    - Gradient norms
    - Task correlation analysis
    """

    def __init__(self, output_dir: Path = Path("outputs/training_vis")):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for metrics
        self.metrics_history: List[Dict] = []
        self.task_names: List[str] = []

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics (train_loss_TaskA, val_loss_TaskB, etc.)
        """
        metrics_with_epoch = {"epoch": epoch, **metrics}
        self.metrics_history.append(metrics_with_epoch)

        # Extract task names from metrics
        for key in metrics.keys():
            if key.startswith("train_loss_") or key.startswith("val_loss_"):
                task_name = key.split("_", 2)[-1]  # Extract task name
                if task_name not in self.task_names and task_name != "loss":
                    self.task_names.append(task_name)

    def plot_task_losses(
        self,
        save_path: Optional[Path] = None,
        show_train: bool = True,
        show_val: bool = True,
    ) -> None:
        """Plot loss curves for all tasks.

        Args:
            save_path: Path to save plot (None = use default)
            show_train: Show training losses
            show_val: Show validation losses
        """
        if not self.metrics_history:
            logger.bind(source="visualizer").warning("No metrics to plot")
            return

        df = pd.DataFrame(self.metrics_history)

        # Create subplots for each task
        num_tasks = len(self.task_names)
        cols = 3
        rows = (num_tasks + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if num_tasks > 1 else [axes]

        for idx, task_name in enumerate(self.task_names):
            ax = axes[idx]

            # Plot train loss
            if show_train and f"train_loss_{task_name}" in df.columns:
                ax.plot(
                    df["epoch"],
                    df[f"train_loss_{task_name}"],
                    label="Train",
                    linewidth=2,
                    alpha=0.8,
                )

            # Plot val loss
            if show_val and f"val_loss_{task_name}" in df.columns:
                ax.plot(
                    df["epoch"],
                    df[f"val_loss_{task_name}"],
                    label="Val",
                    linewidth=2,
                    alpha=0.8,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{task_name} Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_tasks, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "task_losses.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.bind(source="visualizer").info(f"Saved task loss plot to {save_path}")

    def plot_task_weights(self, save_path: Optional[Path] = None) -> None:
        """Plot evolution of task weights over training.

        Args:
            save_path: Path to save plot (None = use default)
        """
        if not self.metrics_history:
            logger.bind(source="visualizer").warning("No metrics to plot")
            return

        df = pd.DataFrame(self.metrics_history)

        # Extract weight columns
        weight_cols = [col for col in df.columns if col.startswith("task_weight_")]

        if not weight_cols:
            logger.bind(source="visualizer").warning("No task weight metrics found")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        for col in weight_cols:
            task_name = col.replace("task_weight_", "")
            ax.plot(df["epoch"], df[col], label=task_name, linewidth=2, alpha=0.8, marker="o")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Task Weight")
        ax.set_title("Task Weight Evolution")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "task_weights.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.bind(source="visualizer").info(f"Saved task weight plot to {save_path}")

    def plot_overall_loss(self, save_path: Optional[Path] = None) -> None:
        """Plot overall train and validation loss.

        Args:
            save_path: Path to save plot (None = use default)
        """
        if not self.metrics_history:
            logger.bind(source="visualizer").warning("No metrics to plot")
            return

        df = pd.DataFrame(self.metrics_history)

        fig, ax = plt.subplots(figsize=(10, 6))

        if "train_loss" in df.columns:
            ax.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2, marker="o")

        if "val_loss" in df.columns:
            ax.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2, marker="o")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Overall Training Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "overall_loss.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.bind(source="visualizer").info(f"Saved overall loss plot to {save_path}")

    def plot_task_correlation(
        self,
        metric: str = "val_loss",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot correlation matrix between task losses.

        High correlation suggests tasks are learning similar patterns.
        Negative correlation might indicate task interference.

        Args:
            metric: Metric to correlate (e.g., "val_loss", "train_loss")
            save_path: Path to save plot (None = use default)
        """
        if not self.metrics_history:
            logger.bind(source="visualizer").warning("No metrics to plot")
            return

        df = pd.DataFrame(self.metrics_history)

        # Extract task-specific metrics
        task_cols = [col for col in df.columns if col.startswith(f"{metric}_")]
        task_cols = [col for col in task_cols if col != f"{metric}"]  # Exclude overall metric

        if len(task_cols) < 2:
            logger.bind(source="visualizer").warning(
                f"Not enough task {metric} columns for correlation"
            )
            return

        # Compute correlation
        task_df = df[task_cols]
        task_df.columns = [col.replace(f"{metric}_", "") for col in task_cols]
        corr = task_df.corr()

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
        )
        ax.set_title(f"Task {metric.replace('_', ' ').title()} Correlation")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"task_{metric}_correlation.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.bind(source="visualizer").info(f"Saved correlation plot to {save_path}")

    def plot_task_comparison(
        self,
        metric: str = "val_loss",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot bar chart comparing final task performance.

        Args:
            metric: Metric to compare (e.g., "val_loss", "val_accuracy")
            save_path: Path to save plot (None = use default)
        """
        if not self.metrics_history:
            logger.bind(source="visualizer").warning("No metrics to plot")
            return

        # Get final epoch metrics
        final_metrics = self.metrics_history[-1]

        # Extract task-specific metrics
        task_metrics = {
            key.replace(f"{metric}_", ""): value
            for key, value in final_metrics.items()
            if key.startswith(f"{metric}_") and key != f"{metric}"
        }

        if not task_metrics:
            logger.bind(source="visualizer").warning(f"No {metric} metrics found")
            return

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        tasks = list(task_metrics.keys())
        values = list(task_metrics.values())

        bars = ax.bar(range(len(tasks)), values, alpha=0.7, edgecolor="black")

        # Color bars by performance (lower is better for loss, higher for accuracy)
        if "loss" in metric:
            colors = plt.cm.RdYlGn_r(np.array(values) / max(values))  # Red = high loss (bad)
        else:
            colors = plt.cm.RdYlGn(np.array(values) / max(values))  # Green = high acc (good)

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Final Task Performance ({metric})")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"task_{metric}_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.bind(source="visualizer").info(f"Saved comparison plot to {save_path}")

    def export_metrics(
        self,
        save_path: Optional[Path] = None,
        format: str = "csv",
    ) -> None:
        """Export metrics to file for external analysis.

        Args:
            save_path: Path to save metrics (None = use default)
            format: Export format ("csv", "json", "parquet")
        """
        if not self.metrics_history:
            logger.bind(source="visualizer").warning("No metrics to export")
            return

        df = pd.DataFrame(self.metrics_history)

        if save_path is None:
            save_path = self.output_dir / f"training_metrics.{format}"

        if format == "csv":
            df.to_csv(save_path, index=False)
        elif format == "json":
            df.to_json(save_path, orient="records", indent=2)
        elif format == "parquet":
            df.to_parquet(save_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.bind(source="visualizer").info(f"Exported metrics to {save_path}")

    def generate_training_report(self) -> Dict:
        """Generate a comprehensive training report.

        Returns:
            Dict with training statistics and analysis
        """
        if not self.metrics_history:
            return {"error": "No metrics available"}

        df = pd.DataFrame(self.metrics_history)

        report = {
            "total_epochs": len(df),
            "task_count": len(self.task_names),
            "tasks": self.task_names,
            "final_metrics": self.metrics_history[-1],
            "best_val_loss": {},
            "convergence_epoch": {},
        }

        # Find best validation loss for each task
        for task in self.task_names:
            val_loss_col = f"val_loss_{task}"
            if val_loss_col in df.columns:
                best_idx = df[val_loss_col].idxmin()
                report["best_val_loss"][task] = {
                    "loss": float(df.loc[best_idx, val_loss_col]),
                    "epoch": int(df.loc[best_idx, "epoch"]),
                }

                # Estimate convergence (when loss stops improving significantly)
                losses = df[val_loss_col].values
                # Find first epoch where improvement is < 1% for 5 consecutive epochs
                for i in range(5, len(losses)):
                    recent_improvement = (losses[i-5] - losses[i]) / (losses[i-5] + 1e-8)
                    if recent_improvement < 0.01:
                        report["convergence_epoch"][task] = int(df.loc[i, "epoch"])
                        break

        return report

    def create_all_plots(self) -> None:
        """Generate all available plots."""
        logger.bind(source="visualizer").info("Generating all training visualizations...")

        self.plot_overall_loss()
        self.plot_task_losses()
        self.plot_task_weights()
        self.plot_task_correlation(metric="val_loss")
        self.plot_task_correlation(metric="train_loss")
        self.plot_task_comparison(metric="val_loss")

        # Export metrics
        self.export_metrics(format="csv")
        self.export_metrics(format="json")

        # Generate report
        report = self.generate_training_report()
        report_path = self.output_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.bind(source="visualizer").info(
            f"Generated all plots and report in {self.output_dir}"
        )


def plot_ewc_importance(
    ewc_importance: Dict[str, float],
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> None:
    """Plot parameter importance from EWC Fisher Information.

    Args:
        ewc_importance: Dict mapping parameter name -> importance score
        top_n: Number of top parameters to show
        save_path: Path to save plot
    """
    if not ewc_importance:
        logger.bind(source="visualizer").warning("No EWC importance data")
        return

    # Sort by importance
    sorted_params = sorted(ewc_importance.items(), key=lambda x: x[1], reverse=True)
    top_params = sorted_params[:top_n]

    names = [name for name, _ in top_params]
    scores = [score for _, score in top_params]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    bars = ax.barh(range(len(names)), scores, alpha=0.7, edgecolor="black")

    # Color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Fisher Information (Importance)")
    ax.set_title(f"Top {top_n} Most Important Parameters (EWC)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.bind(source="visualizer").info(f"Saved EWC importance plot to {save_path}")

    plt.close()


__all__ = ["TrainingVisualizer", "plot_ewc_importance"]
