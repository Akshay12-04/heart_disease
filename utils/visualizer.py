"""
=============================================================
  visualizer.py  –  All plotting utilities
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

PALETTE = {
    "federated"  : "#2196F3",
    "centralized": "#FF5722",
    "accent"     : "#4CAF50",
    "warn"       : "#FFC107",
    "bg"         : "#F5F5F5"
}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Visualizer] Saved → {path}")


# ── 1. Training curves ────────────────────────────────────────────────────────
def plot_training_curves(round_logs, save_path):
    rounds   = [r["round"] for r in round_logs]
    fed_loss = [r["avg_client_loss"] for r in round_logs]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Federated Learning – Training Progress", fontsize=15, fontweight="bold")
    ax.plot(rounds, fed_loss, color=PALETTE["warn"], linewidth=2.5, marker="o", markersize=5)
    ax.fill_between(rounds, fed_loss, alpha=0.1, color=PALETTE["warn"])
    ax.set_title("Average Client Loss", fontsize=12)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(min(rounds)-0.5, max(rounds)+0.5)

    plt.tight_layout()
    _save(fig, save_path)


# ── 2. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Heart Disease"],
                yticklabels=["No Disease", "Heart Disease"],
                ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    _save(fig, save_path)


# ── 3. Federated vs Centralised ───────────────────────────────────────────────
def plot_federated_vs_centralized(fed_metrics, cen_metrics, save_path):
    metrics  = ["accuracy", "precision", "recall", "f1"]
    labels   = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fed_vals = [fed_metrics[m] for m in metrics]
    cen_vals = [cen_metrics[m] for m in metrics]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1  = ax.bar(x - w/2, fed_vals, w, label="Federated",   color=PALETTE["federated"], alpha=0.85)
    b2  = ax.bar(x + w/2, cen_vals, w, label="Centralized", color=PALETTE["centralized"], alpha=0.85)

    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_title("Federated vs Centralised Learning", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, save_path)


# ── 4. Per-client accuracy bars ───────────────────────────────────────────────
def plot_client_accuracy(clients_val_metrics, save_path):
    ids  = [f"H-{m['id']:02d}" for m in clients_val_metrics]
    accs = [m["accuracy"] for m in clients_val_metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [PALETTE["federated"] if a >= 0.75 else PALETTE["warn"] for a in accs]
    bars    = ax.bar(ids, accs, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_title("Per-Hospital Validation Accuracy (Final Round)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hospital")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=np.mean(accs), color="red", linestyle="--", linewidth=1.5, label=f"Mean={np.mean(accs):.3f}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, save_path)


# ── 5. Metric summary table ───────────────────────────────────────────────────
def plot_metrics_table(fed_metrics, cen_metrics, save_path):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fed_row = [f"{fed_metrics[m]:.4f}" for m in metrics]
    cen_row = [f"{cen_metrics[m]:.4f}" for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis("off")
    table   = ax.table(
        cellText=[fed_row, cen_row],
        rowLabels=["  Federated  ", "  Centralized  "],
        colLabels=labels,
        cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color header
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold")

    ax.set_title("Model Performance Summary", fontsize=12, fontweight="bold", pad=20)
    _save(fig, save_path)


# ── 6. Feature importance (permutation proxy) ────────────────────────────────
def plot_feature_importance(model, X_test, y_test, feature_names, save_path, device="cpu"):
    import torch
    from sklearn.metrics import accuracy_score

    model.eval()
    base_metrics = _quick_acc(model, X_test, y_test, device)
    importances  = []
    for i in range(X_test.shape[1]):
        X_perm = X_test.copy()
        np.random.shuffle(X_perm[:, i])
        acc_perm = _quick_acc(model, X_perm, y_test, device)
        importances.append(base_metrics - acc_perm)

    idx  = np.argsort(importances)[-15:]  # top-15
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = [PALETTE["accent"] if v >= 0 else PALETTE["warn"] for v in np.array(importances)[idx]]
    ax.barh([feature_names[i] for i in idx], [importances[i] for i in idx], color=colors)
    ax.set_title("Feature Importance (Permutation, Top-15)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Drop in Accuracy when feature is shuffled")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    _save(fig, save_path)


def _quick_acc(model, X, y, device):
    import torch
    model.to(device)
    with torch.no_grad():
        preds = (model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy() >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score
    return accuracy_score(y, preds)
