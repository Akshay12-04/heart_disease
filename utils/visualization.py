"""
=============================================================================
Federated Learning for Heart Disease Prediction
File: utils/visualization.py
Purpose: All plotting — training curves, confusion matrix, FL vs centralised
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os


PALETTE = {
    "federated":    "#2196F3",
    "centralised":  "#FF5722",
    "accent":       "#4CAF50",
    "grid":         "#e0e0e0",
    "bg":           "#fafafa",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["bg"],
        "axes.grid":        True,
        "grid.color":       PALETTE["grid"],
        "grid.linewidth":   0.6,
        "font.family":      "DejaVu Sans",
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curves (loss + accuracy per round)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(fl_history: list, save_path: str):
    """
    fl_history: list of dicts from FederatedServer.history
    Each dict has keys: round, avg_loss, avg_acc
    """
    _style()
    rounds    = [h["round"]    for h in fl_history]
    losses    = [h["avg_loss"] for h in fl_history]
    accs      = [h["avg_acc"]  for h in fl_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Federated Learning — Training Progress", fontsize=15, fontweight="bold")

    # Loss curve
    ax1.plot(rounds, losses, marker="o", color=PALETTE["federated"], linewidth=2, markersize=5)
    ax1.fill_between(rounds, losses, alpha=0.15, color=PALETTE["federated"])
    ax1.set_title("Average Training Loss per Round")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Loss (BCE)")
    ax1.set_xticks(rounds)

    # Accuracy curve
    ax2.plot(rounds, [a * 100 for a in accs], marker="s",
             color=PALETTE["accent"], linewidth=2, markersize=5)
    ax2.fill_between(rounds, [a * 100 for a in accs], alpha=0.15, color=PALETTE["accent"])
    ax2.set_title("Average Training Accuracy per Round")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(rounds)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Training curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-client accuracy across rounds
# ─────────────────────────────────────────────────────────────────────────────
def plot_client_accuracies(clients: list, save_path: str):
    """Plot each hospital's local test accuracy across FL rounds."""
    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Per-Hospital Local Test Accuracy Across Rounds",
                 fontsize=14, fontweight="bold")

    cmap = plt.cm.get_cmap("tab10", len(clients))

    for i, client in enumerate(clients):
        if client.test_accs:
            rounds = list(range(1, len(client.test_accs) + 1))
            ax.plot(rounds, [a * 100 for a in client.test_accs],
                    marker="o", label=client.hospital_id,
                    color=cmap(i), linewidth=2)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Client accuracy plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    """Plot a 2×2 confusion matrix with value annotations."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["No Disease (0)", "Heart Disease (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Label");  ax.set_ylabel("True Label")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FL vs Centralised comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_fl_vs_centralised(fl_metrics: dict, central_metrics: dict, save_path: str):
    """Side-by-side bar chart comparing FL vs centralised training."""
    _style()
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    metric_keys  = ["accuracy", "precision", "recall", "f1", "auc_roc"]

    fl_vals  = [fl_metrics.get(k, 0)      for k in metric_keys]
    cen_vals = [central_metrics.get(k, 0) for k in metric_keys]

    x     = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Federated Learning vs Centralised Learning", fontsize=14, fontweight="bold")

    bars1 = ax.bar(x - width/2, fl_vals,  width, label="Federated Learning",  color=PALETTE["federated"],   alpha=0.85)
    bars2 = ax.bar(x + width/2, cen_vals, width, label="Centralised Learning", color=PALETTE["centralised"], alpha=0.85)

    # Value labels on top of bars
    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] FL vs Centralised comparison saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metrics dashboard (summary card)
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_dashboard(fl_metrics: dict, central_metrics: dict, save_path: str):
    """
    A combined 2×2 figure: training curves + confusion matrices + comparison.
    Used as the headline summary figure in the README / report.
    """
    _style()
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Federated Learning — Final Evaluation Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    metric_keys  = ["accuracy", "precision", "recall", "f1", "auc_roc"]

    # ── Panel 1: FL metrics ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    vals = [fl_metrics.get(k, 0) for k in metric_keys]
    bars = ax1.barh(metric_names, vals, color=PALETTE["federated"], alpha=0.85)
    ax1.set_title("Federated Learning Metrics")
    ax1.set_xlim(0, 1.1)
    for bar, v in zip(bars, vals):
        ax1.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=10)

    # ── Panel 2: Centralised metrics ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    vals2 = [central_metrics.get(k, 0) for k in metric_keys]
    bars2 = ax2.barh(metric_names, vals2, color=PALETTE["centralised"], alpha=0.85)
    ax2.set_title("Centralised Learning Metrics")
    ax2.set_xlim(0, 1.1)
    for bar, v in zip(bars2, vals2):
        ax2.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=10)

    # ── Panel 3: Delta ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    deltas = [f - c for f, c in zip(vals, vals2)]
    colors = [PALETTE["accent"] if d >= 0 else PALETTE["centralised"] for d in deltas]
    ax3.barh(metric_names, deltas, color=colors, alpha=0.85)
    ax3.axvline(0, color="black", linewidth=1)
    ax3.set_title("Δ (Federated − Centralised)")
    for i, d in enumerate(deltas):
        ax3.text(d + (0.002 if d >= 0 else -0.002), i,
                 f"{d:+.3f}", va="center",
                 ha="left" if d >= 0 else "right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Dashboard saved → {save_path}")
