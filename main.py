"""
=============================================================
  main.py  –  Entry point for the full FL pipeline

  Run:
      python main.py
      python main.py --hospitals 5 --rounds 15 --iid
      python main.py --hospitals 5 --rounds 15 --noniid --dp
=============================================================
"""

import argparse
import sys, os
import json
import random
import numpy as np
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.data_preprocessor import (load_and_preprocess,
                                     split_into_hospital_partitions,
                                     split_train_and_global_test,
                                     fit_scaler_on_train, save_scaler)
from models.heart_model import evaluate_model
from clients.hospital_client import HospitalClient
from server.federated_server import FederatedServer
from utils.centralized_baseline import train_centralized
from utils.visualizer import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_feature_importance,
)


# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Federated Learning – Heart Disease")
    p.add_argument("--hospitals", type=int, default=5,  help="Number of hospital clients")
    p.add_argument("--rounds",    type=int, default=10, help="FL communication rounds")
    p.add_argument("--epochs",    type=int, default=5,  help="Local epochs per round")
    p.add_argument("--noniid",    action="store_true",  help="Non-IID data split")
    p.add_argument("--dp",        action="store_true",  help="Enable Differential Privacy")
    p.add_argument("--fraction",  type=float, default=1.0, help="Client fraction per round")
    p.add_argument("--seed",      type=int, default=42, help="Random seed for reproducible runs")
    return p.parse_args()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    if args.hospitals < 2:
        raise ValueError("--hospitals must be >= 2")
    if args.rounds < 1 or args.epochs < 1:
        raise ValueError("--rounds and --epochs must be >= 1")
    if not (0.0 < args.fraction <= 1.0):
        raise ValueError("--fraction must be in (0, 1]")
    set_global_seed(args.seed)

    print("\n" + "█"*60)
    print("  FEDERATED LEARNING FOR HEART DISEASE PREDICTION")
    print("  Final Year Engineering Project")
    print("█"*60)
    print(f"\n  Config → Hospitals={args.hospitals}  Rounds={args.rounds}  "
          f"Epochs={args.epochs}  IID={'No' if args.noniid else 'Yes'}  "
          f"DP={'On' if args.dp else 'Off'}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n")

    # ── 1. Load & preprocess ──────────────────────────────────────────────
    data_path = os.path.join(BASE_DIR, "data", "heart_disease_federated.csv")
    X, y, feature_names = load_and_preprocess(data_path)

    # Hold-out global test set (never seen by any client)
    X_train_pool, X_test, y_train_pool, y_test = split_train_and_global_test(
        X, y, test_size=0.15, seed=args.seed
    )
    X_train_pool, X_test, scaler = fit_scaler_on_train(X_train_pool, X_test)
    print(f"\n  Global test set: {len(X_test)} samples\n")

    # Save scaler for Streamlit inference
    save_scaler(scaler, os.path.join(BASE_DIR, "models", "scaler.pkl"))

    input_dim = X.shape[1]

    # ── 2. Partition data across hospitals ────────────────────────────────
    print(f"  Partitioning data → {args.hospitals} hospitals "
          f"({'Non-IID' if args.noniid else 'IID'}):\n")
    partitions = split_into_hospital_partitions(
        X_train_pool, y_train_pool,
        num_hospitals=args.hospitals,
        iid=not args.noniid,
        seed=args.seed,
    )

    # ── 3. Create hospital clients ────────────────────────────────────────
    clients = [
        HospitalClient(
            hospital_id  = p["hospital_id"],
            X_train      = p["X_train"],
            y_train      = p["y_train"],
            X_val        = p["X_val"],
            y_val        = p["y_val"],
            input_dim    = input_dim,
            device       = device,
            local_epochs = args.epochs,
            dp_enabled   = args.dp,
            dp_noise_multiplier=0.01
        )
        for p in partitions
    ]
    print(f"\n  {len(clients)} HospitalClients created:\n")
    for c in clients:
        print(f"    {c}")

    # ── 4. Initialise server & run FL ─────────────────────────────────────
    server = FederatedServer(input_dim=input_dim, device=device)
    round_logs = server.run(clients, X_test, y_test,
                            num_rounds=args.rounds,
                            fraction_clients=args.fraction)

    # ── 5. Save global model ──────────────────────────────────────────────
    model_path = os.path.join(BASE_DIR, "models", "global_model.pth")
    server.save_global_model(model_path)
    server.save_logs(os.path.join(BASE_DIR, "results", "metrics", "round_logs.json"))

    # ── 6. Final global evaluation ────────────────────────────────────────
    fed_metrics = evaluate_model(server.global_model, X_test, y_test, device)
    print("\n" + "─"*50)
    print("  FINAL FEDERATED MODEL METRICS")
    print("─"*50)
    print(f"  Confusion Matrix:\n  {np.array(fed_metrics['confusion_matrix'])}\n")

    # ── 7. Centralised baseline ────────────────────────────────────────────
    # Use FULL training portion (same data, but pooled)
    X_train_full = np.concatenate([p["X_train"] for p in partitions])
    y_train_full = np.concatenate([p["y_train"] for p in partitions])
    cen_metrics, _, _ = train_centralized(
        X_train_full, y_train_full, X_test, y_test,
        epochs=args.rounds * args.epochs, device=device
    )

    # ── 8. Plots ───────────────────────────────────────────────────────────
    plots_dir = os.path.join(BASE_DIR, "results", "plots")
    print("\n  Generating plots …")

    for plot_name, plot_fn in [
        ("01_training_curves.png", lambda: plot_training_curves(round_logs, os.path.join(plots_dir, "01_training_curves.png"))),
        ("02_fed_confusion_matrix.png", lambda: plot_confusion_matrix(fed_metrics["confusion_matrix"], "Federated Model – Confusion Matrix", os.path.join(plots_dir, "02_fed_confusion_matrix.png"))),
        ("03_cen_confusion_matrix.png", lambda: plot_confusion_matrix(cen_metrics["confusion_matrix"], "Centralised Model – Confusion Matrix", os.path.join(plots_dir, "03_cen_confusion_matrix.png"))),
        ("07_feature_importance.png", lambda: plot_feature_importance(server.global_model, X_test, y_test, feature_names, os.path.join(plots_dir, "07_feature_importance.png"), device)),
    ]:
        try:
            plot_fn()
        except Exception as exc:
            print(f"[Warning] Plot generation failed for {plot_name}: {exc}")

    # ── 9. Save final metrics JSON ────────────────────────────────────────
    summary = {
        "federated": {"confusion_matrix": fed_metrics["confusion_matrix"]},
        "centralized": {"confusion_matrix": cen_metrics["confusion_matrix"]},
        "config"      : vars(args)
    }
    with open(os.path.join(BASE_DIR, "results", "metrics", "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  All results saved to results/")
    print("\n" + "█"*60)
    print("  DONE – Federated Learning pipeline complete!")
    print("█"*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[Error] Pipeline failed: {exc}")
        raise SystemExit(1)
