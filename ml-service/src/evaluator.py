"""
evaluator.py

Full model evaluation suite for the RealTimeGuard
fraud detection pipeline.

Generates:
  - Classification report (text)
  - Confusion matrix heatmap  → reports/confusion_matrix.png
  - ROC curve                 → reports/roc_curve.png
  - Precision-Recall curve    → reports/pr_curve.png
  - Feature importance chart  → reports/feature_importance.png
  - Metrics summary           → reports/metrics.json
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for servers/CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import REPORT_DIR
from src.utils import save_json, setup_logger

logger = setup_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Plot style constants
# ─────────────────────────────────────────────────────────────
_PALETTE = {
    "bg": "#0F1117",
    "surface": "#1A1D27",
    "accent": "#7C3AED",
    "accent_2": "#06B6D4",
    "positive": "#10B981",
    "negative": "#EF4444",
    "text": "#E2E8F0",
    "subtext": "#94A3B8",
    "grid": "#2D3148",
}
_FIG_DPI = 150


def _apply_dark_style() -> None:
    """Apply a consistent dark theme to all matplotlib plots."""
    plt.rcParams.update(
        {
            "figure.facecolor": _PALETTE["bg"],
            "axes.facecolor": _PALETTE["surface"],
            "axes.edgecolor": _PALETTE["grid"],
            "axes.labelcolor": _PALETTE["text"],
            "axes.titlecolor": _PALETTE["text"],
            "xtick.color": _PALETTE["subtext"],
            "ytick.color": _PALETTE["subtext"],
            "grid.color": _PALETTE["grid"],
            "grid.linewidth": 0.6,
            "text.color": _PALETTE["text"],
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
        }
    )


class ModelEvaluator:
    """
    Evaluates a trained fraud detection model and generates
    reports and visualizations.

    Parameters
    ----------
    model : LGBMClassifier
        A fitted LightGBM model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True binary labels.
    report_dir : Path, optional
        Directory for output files. Defaults to REPORT_DIR from config.
    """

    def __init__(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        report_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.report_dir = Path(report_dir) if report_dir else REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Populated after evaluate()
        self.y_pred: np.ndarray | None = None
        self.y_prob: np.ndarray | None = None
        self.metrics: dict = {}

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def evaluate(self) -> dict:
        """
        Compute and return all evaluation metrics.

        Returns
        -------
        dict
            Keys: accuracy, precision, recall, f1_score, roc_auc,
                  confusion_matrix (TP/TN/FP/FN), classification_report.
        """
        self.y_pred = self.model.predict(self.X_test)
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]

        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.metrics = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred),
            "recall": recall_score(self.y_test, self.y_pred),
            "f1_score": f1_score(self.y_test, self.y_pred),
            "roc_auc": roc_auc_score(self.y_test, self.y_prob),
            "confusion_matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            },
        }

        logger.info("─" * 50)
        logger.info("Evaluation Results")
        logger.info("─" * 50)
        for key, val in self.metrics.items():
            if key == "confusion_matrix":
                logger.info("  %-20s  %s", key, val)
            else:
                logger.info("  %-20s  %.6f", key, val)
        logger.info("─" * 50)

        logger.info("\nClassification Report:\n%s",
                    classification_report(self.y_test, self.y_pred,
                                          target_names=["Normal", "Fraud"]))

        return self.metrics

    def generate_reports(self) -> None:
        """
        Generate and save all visual evaluation reports.
        Must be called after evaluate().
        """
        if self.y_pred is None:
            raise RuntimeError("Call evaluate() before generate_reports().")

        _apply_dark_style()

        self._plot_confusion_matrix()
        self._plot_roc_curve()
        self._plot_pr_curve()
        self._plot_feature_importance()
        self._save_metrics_json()

        logger.info("All reports saved to → %s", self.report_dir)

    # --------------------------------------------------
    # Private plot methods
    # --------------------------------------------------

    def _plot_confusion_matrix(self) -> None:
        cm = confusion_matrix(self.y_test, self.y_pred)
        labels = ["Normal", "Fraud"]

        fig, ax = plt.subplots(figsize=(6, 5), dpi=_FIG_DPI)
        fig.patch.set_facecolor(_PALETTE["bg"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.light_palette(_PALETTE["accent"], as_cmap=True),
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
            linecolor=_PALETTE["grid"],
            ax=ax,
            annot_kws={"size": 14, "weight": "bold", "color": _PALETTE["text"]},
        )

        ax.set_title("Confusion Matrix", pad=15)
        ax.set_xlabel("Predicted Label", labelpad=10)
        ax.set_ylabel("True Label", labelpad=10)

        # Style colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=_PALETTE["subtext"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_PALETTE["subtext"])

        plt.tight_layout()
        out = self.report_dir / "confusion_matrix.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Saved → %s", out)

    def _plot_roc_curve(self) -> None:
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=_FIG_DPI)
        fig.patch.set_facecolor(_PALETTE["bg"])

        ax.plot(
            fpr, tpr,
            color=_PALETTE["accent"],
            linewidth=2.5,
            label=f"ROC Curve  (AUC = {roc_auc:.4f})",
        )
        ax.plot(
            [0, 1], [0, 1],
            color=_PALETTE["subtext"],
            linewidth=1,
            linestyle="--",
            label="Random Classifier",
        )
        ax.fill_between(fpr, tpr, alpha=0.08, color=_PALETTE["accent"])

        ax.set_title("ROC Curve — Fraud Detection", pad=15)
        ax.set_xlabel("False Positive Rate", labelpad=10)
        ax.set_ylabel("True Positive Rate", labelpad=10)
        ax.legend(
            frameon=True,
            facecolor=_PALETTE["surface"],
            edgecolor=_PALETTE["grid"],
            labelcolor=_PALETTE["text"],
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])

        plt.tight_layout()
        out = self.report_dir / "roc_curve.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Saved → %s", out)

    def _plot_pr_curve(self) -> None:
        precision_vals, recall_vals, _ = precision_recall_curve(
            self.y_test, self.y_prob
        )
        pr_auc = auc(recall_vals, precision_vals)

        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=_FIG_DPI)
        fig.patch.set_facecolor(_PALETTE["bg"])

        ax.plot(
            recall_vals, precision_vals,
            color=_PALETTE["accent_2"],
            linewidth=2.5,
            label=f"PR Curve  (AUC = {pr_auc:.4f})",
        )
        ax.fill_between(recall_vals, precision_vals, alpha=0.08,
                        color=_PALETTE["accent_2"])

        # Baseline — random classifier
        baseline = (self.y_test == 1).sum() / len(self.y_test)
        ax.axhline(
            y=baseline,
            color=_PALETTE["subtext"],
            linewidth=1,
            linestyle="--",
            label=f"Random Baseline ({baseline:.4f})",
        )

        ax.set_title("Precision–Recall Curve — Fraud Detection", pad=15)
        ax.set_xlabel("Recall", labelpad=10)
        ax.set_ylabel("Precision", labelpad=10)
        ax.legend(
            frameon=True,
            facecolor=_PALETTE["surface"],
            edgecolor=_PALETTE["grid"],
            labelcolor=_PALETTE["text"],
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])

        plt.tight_layout()
        out = self.report_dir / "pr_curve.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Saved → %s", out)

    def _plot_feature_importance(self) -> None:
        importances = self.model.feature_importances_
        feature_names = self.X_test.columns.tolist()

        # Sort descending
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_vals = importances[indices]

        # Colour: top 3 in accent, rest in accent_2
        colors = [
            _PALETTE["accent"] if i < 3 else _PALETTE["accent_2"]
            for i in range(len(sorted_names))
        ]

        fig, ax = plt.subplots(figsize=(9, 6), dpi=_FIG_DPI)
        fig.patch.set_facecolor(_PALETTE["bg"])

        bars = ax.barh(
            sorted_names[::-1],
            sorted_vals[::-1],
            color=colors[::-1],
            edgecolor="none",
            height=0.7,
        )

        # Value labels on bars
        for bar, val in zip(bars, sorted_vals[::-1]):
            ax.text(
                bar.get_width() + max(sorted_vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}",
                va="center",
                color=_PALETTE["subtext"],
                fontsize=9,
            )

        ax.set_title("Feature Importance (LightGBM Gain)", pad=15)
        ax.set_xlabel("Importance Score", labelpad=10)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out = self.report_dir / "feature_importance.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Saved → %s", out)

    def _save_metrics_json(self) -> None:
        out = self.report_dir / "metrics.json"
        save_json(self.metrics, out)
        logger.info("Saved → %s", out)
