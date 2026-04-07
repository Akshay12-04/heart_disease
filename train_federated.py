"""
=============================================================================
Federated Learning for Heart Disease Prediction
File: train_federated.py
Purpose: Main training script — orchestrates FL rounds end-to-end
=============================================================================

Run:
    python train_federated.py

Optional flags:
    --hospitals   5        Number of hospital clients (default: 5)
    --rounds      15       Number of FL communication rounds (default: 15)
    --epochs      5        Local epochs per hospital per round (default: 5)
    --lr          0.001    Learning rate (default: 0.001)
    --dp                   Enable differential privacy (flag, default: off)
    --noniid               Use non-IID data split (flag, default: IID)
=============================================================================
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch

# ── Project path setup ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from utils.data_preprocessing import full_pipeline
from models.heart_model import HeartDiseaseNet, set_weights, evaluate
from clients.hospital_client import create_hospital_clients
from server.federated_server import FederatedServer
from utils.visualization import (
    plot_training_curves,
    plot_client_accuracies,
    plot_confusion_matrix,
    plot_fl_vs_centralised,
    plot_metrics_dashboard,
)


# ─────────────────────────────────────────────────────────────────────────────
# CENTRALISED BASELINE (train on all data combined, no FL)
# ─────────────────────────────────────────────────────────────────────────────
def train_centralised_baseline(
    hospital_data: list,
    X_global_test: np.ndarray,
    y_global_test: np.ndarray,
    input_dim: int,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """
    Combines all hospital training data (as if privacy didn't exist)
    to get a baseline we can compare federated learning against.
    """
    from models.heart_model import local_train

    print("\n" + "="*60)
    print("  CENTRALISED TRAINING BASELINE")
    print("="*60)

    # Merge all hospital data
    X_all = np.vstack([h["X_train"] for h in hospital_data])
    y_all = np.concatenate([h["y_train"] for h in hospital_data])
    print(f"  Combined training set: {len(X_all)} samples")

    model = HeartDiseaseNet(input_dim=input_dim).to(device)
    model, loss, acc = local_train(
        model, X_all, y_all,
        epochs=epochs, lr=lr, batch_size=64, device=device,
    )
    metrics = evaluate(model, X_global_test, y_global_test, device)

    print(f"  Centralised  accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Centralised  F1-score  : {metrics['f1']:.4f}")
    print(f"  Centralised  AUC-ROC   : {metrics['auc_roc']:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_federated(
    hospital_data: list,
    X_global_test: np.ndarray,
    y_global_test: np.ndarray,
    input_dim: int,
    n_rounds: int = 15,
    local_epochs: int = 5,
    lr: float = 1e-3,
    use_dp: bool = False,
    results_dir: str = None,
    device: str = "cpu",
):
    """
    Federated Learning training loop.

    Communication Round sequence (per round):
      1. Server → all clients: broadcast global weights
      2. Each client: train locally for `local_epochs`
      3. Each client → server: send updated weights
      4. Server: FedAvg aggregation → new global model
      5. Server evaluates global model on held-out test set
    """
    print("\n" + "="*60)
    print("  FEDERATED LEARNING TRAINING")
    print(f"  Hospitals={len(hospital_data)} | Rounds={n_rounds} | "
          f"Local epochs={local_epochs} | DP={use_dp}")
    print("="*60)

    # ── Initialise server ────────────────────────────────────────────────────
    server = FederatedServer(
        input_dim=input_dim,
        use_differential_privacy=use_dp,
        noise_multiplier=0.005,
        save_dir=results_dir,
    )

    # ── Create hospital clients ───────────────────────────────────────────────
    clients = create_hospital_clients(
        hospital_data=hospital_data,
        input_dim=input_dim,
        local_epochs=local_epochs,
        lr=lr,
        device=device,
    )

    # ── Round-level metrics (for global test evaluation) ─────────────────────
    global_eval_history = []

    t_start = time.time()

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'─'*55}")
        print(f"  Communication Round {round_num}/{n_rounds}")
        print(f"{'─'*55}")

        # ── Step 1: Server broadcasts current global weights ─────────────────
        global_weights = server.get_global_weights()

        # ── Step 2: Each hospital receives weights and trains locally ─────────
        client_updates = []
        for client in clients:
            client.receive_global_weights(global_weights)     # download
            update = client.train_locally()                    # local train
            local_eval = client.evaluate_local()               # local eval

            print(f"    {client.hospital_id:12s} | "
                  f"loss={update['loss']:.4f}  "
                  f"train_acc={update['accuracy']*100:.1f}%  "
                  f"test_acc={local_eval['accuracy']*100:.1f}%")

            client_updates.append(update)

        # ── Step 3: Server aggregates ─────────────────────────────────────────
        server.aggregate(client_updates)

        # ── Step 4: Evaluate global model on hold-out test set ────────────────
        set_weights(server.global_model, server.get_global_weights())
        g_metrics = evaluate(server.global_model, X_global_test, y_global_test, device)

        global_eval_history.append({
            "round":    round_num,
            "accuracy": g_metrics["accuracy"],
            "f1":       g_metrics["f1"],
            "auc_roc":  g_metrics["auc_roc"],
        })

        print(f"\n  🌐 Global Model | "
              f"acc={g_metrics['accuracy']*100:.2f}%  "
              f"f1={g_metrics['f1']:.4f}  "
              f"auc={g_metrics['auc_roc']:.4f}")

    elapsed = time.time() - t_start
    print(f"\n[FL] Training complete in {elapsed:.1f}s")

    # ── Final evaluation ──────────────────────────────────────────────────────
    final_metrics = evaluate(server.global_model, X_global_test, y_global_test, device)

    return server, clients, final_metrics, global_eval_history


# ─────────────────────────────────────────────────────────────────────────────
# PRINT FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(fl_metrics: dict, central_metrics: dict):
    print("\n" + "="*60)
    print("  FINAL EVALUATION REPORT")
    print("="*60)
    header = f"{'Metric':<18} {'Federated':>14} {'Centralised':>14} {'Δ':>10}"
    print(header)
    print("-" * 60)

    for name, key in [
        ("Accuracy",  "accuracy"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1-Score",  "f1"),
        ("AUC-ROC",   "auc_roc"),
    ]:
        fl_v  = fl_metrics.get(key, 0)
        cen_v = central_metrics.get(key, 0)
        delta = fl_v - cen_v
        arrow = "↑" if delta >= 0 else "↓"
        print(f"  {name:<16} {fl_v:>14.4f} {cen_v:>14.4f} {arrow}{abs(delta):>8.4f}")

    print("="*60)
    cm = fl_metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix (Federated Model on Global Test Set):")
    print(f"           Pred=0   Pred=1")
    print(f"  True=0   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"  True=1   {cm[1,0]:6d}   {cm[1,1]:6d}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Federated Learning — Heart Disease")
    parser.add_argument("--hospitals", type=int,   default=5)
    parser.add_argument("--rounds",    type=int,   default=15)
    parser.add_argument("--epochs",    type=int,   default=5)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--dp",        action="store_true", help="Enable differential privacy")
    parser.add_argument("--noniid",    action="store_true", help="Non-IID data split")
    args = parser.parse_args()

    # Paths
    DATA_PATH    = os.path.join(ROOT, "data", "heart_disease_federated.csv")
    ARTIFACTS    = os.path.join(ROOT, "results", "artifacts")
    RESULTS_DIR  = os.path.join(ROOT, "results")
    PLOT_DIR     = os.path.join(ROOT, "results", "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Device: {device}")

    # ── Data pipeline ──────────────────────────────────────────────────────
    print("\n[Step 1] Loading & preprocessing data...")
    hospital_data, X_test, y_test, input_dim, feature_names = full_pipeline(
        filepath     = DATA_PATH,
        n_hospitals  = args.hospitals,
        iid          = not args.noniid,
        artifacts_dir= ARTIFACTS,
    )

    # ── Centralised baseline ────────────────────────────────────────────────
    print("\n[Step 2] Training centralised baseline...")
    central_metrics = train_centralised_baseline(
        hospital_data, X_test, y_test, input_dim,
        epochs=50, lr=args.lr, device=device,
    )

    # ── Federated training ──────────────────────────────────────────────────
    print("\n[Step 3] Federated Learning training...")
    server, clients, fl_metrics, global_hist = train_federated(
        hospital_data  = hospital_data,
        X_global_test  = X_test,
        y_global_test  = y_test,
        input_dim      = input_dim,
        n_rounds       = args.rounds,
        local_epochs   = args.epochs,
        lr             = args.lr,
        use_dp         = args.dp,
        results_dir    = RESULTS_DIR,
        device         = device,
    )

    # ── Save global model ───────────────────────────────────────────────────
    model_path = os.path.join(RESULTS_DIR, "global_model.pt")
    server.save_final_model(model_path)

    # ── Report ──────────────────────────────────────────────────────────────
    print_report(fl_metrics, central_metrics)

    # ── Visualisations ──────────────────────────────────────────────────────
    print("\n[Step 4] Generating plots...")

    plot_training_curves(
        server.history,
        save_path=os.path.join(PLOT_DIR, "training_curves.png"),
    )
    plot_client_accuracies(
        clients,
        save_path=os.path.join(PLOT_DIR, "client_accuracies.png"),
    )
    plot_confusion_matrix(
        fl_metrics["confusion_matrix"],
        title="Federated Model — Confusion Matrix",
        save_path=os.path.join(PLOT_DIR, "confusion_matrix_fl.png"),
    )
    plot_confusion_matrix(
        central_metrics["confusion_matrix"],
        title="Centralised Model — Confusion Matrix",
        save_path=os.path.join(PLOT_DIR, "confusion_matrix_central.png"),
    )
    plot_fl_vs_centralised(
        fl_metrics, central_metrics,
        save_path=os.path.join(PLOT_DIR, "fl_vs_centralised.png"),
    )
    plot_metrics_dashboard(
        fl_metrics, central_metrics,
        save_path=os.path.join(PLOT_DIR, "dashboard.png"),
    )

    # ── Save metrics JSON ───────────────────────────────────────────────────
    def serialise(m):
        return {k: (v.tolist() if hasattr(v, "tolist") else float(v))
                for k, v in m.items() if k != "probabilities" and k != "predictions"}

    results_json = {
        "config":        vars(args),
        "fl_metrics":    serialise(fl_metrics),
        "central_metrics": serialise(central_metrics),
        "global_eval_history": global_hist,
    }
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[Done] Results saved → {json_path}")
    print(f"[Done] Plots saved   → {PLOT_DIR}")
    print(f"[Done] Model saved   → {model_path}")


if __name__ == "__main__":
    main()
