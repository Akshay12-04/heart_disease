"""
=============================================================
  federated_server.py
  Central Federated Learning server.

  Algorithm: FedAvg (McMahan et al., 2017)
  ─────────────────────────────────────────
  For each communication round r:
    1.  Broadcast current global weights w_r to all clients.
    2.  Each client k trains locally → returns w_k, n_k.
    3.  Server aggregates:
              w_{r+1} = Σ_k (n_k / N) × w_k
        where N = Σ n_k  (weighted by local dataset size)
    4.  Evaluate w_{r+1} on held-out global test set.
    5.  Repeat for R rounds.

  The server NEVER sees raw patient data—only weight tensors.
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import json
import time
from copy import deepcopy

from models.heart_model import HeartDiseaseNet, get_parameters, set_parameters, evaluate_model


class FederatedServer:
    def __init__(self, input_dim: int, device: str = "cpu"):
        self.input_dim    = input_dim
        self.device       = device
        self.global_model = HeartDiseaseNet(input_dim)
        self.round_logs   = []   # history for plotting

    # ── Broadcast global weights ────────────────────────────────────────────
    def get_global_parameters(self):
        return get_parameters(self.global_model)

    # ── FedAvg aggregation ──────────────────────────────────────────────────
    def aggregate(self, client_updates):
        """
        FedAvg: weighted average of client model parameters.

        Parameters
        ----------
        client_updates : list of (params, n_samples)
            params    – list of numpy arrays (one per layer)
            n_samples – number of training samples at that client
        """
        total_samples = sum(n for _, n in client_updates)
        global_params = get_parameters(self.global_model)

        # Initialise accumulator with zeros
        new_params = [np.zeros_like(p) for p in global_params]

        for params, n in client_updates:
            weight = n / total_samples
            for i, p in enumerate(params):
                new_params[i] += weight * p

        set_parameters(self.global_model, new_params)

    # ── Run full federated training loop ────────────────────────────────────
    def run(self, clients, X_test, y_test,
            num_rounds: int = 10, fraction_clients: float = 1.0):
        """
        Main training loop.

        fraction_clients : 0–1  fraction of clients selected per round
                           (1.0 = all clients every round)
        """
        print("\n" + "═"*60)
        print("  FEDERATED LEARNING – TRAINING START")
        print(f"  Hospitals : {len(clients)}")
        print(f"  Rounds    : {num_rounds}")
        print(f"  FedAvg fraction: {fraction_clients}")
        print("═"*60 + "\n")

        n_selected = max(1, int(len(clients) * fraction_clients))

        for rnd in range(1, num_rounds + 1):
            t0 = time.time()
            print(f"┌── Round {rnd}/{num_rounds} {'─'*40}")

            # 1. Select clients (random subset if fraction < 1)
            selected = np.random.choice(clients, n_selected, replace=False)

            # 2. Broadcast global weights; collect local updates
            global_params  = self.get_global_parameters()
            client_updates = []
            round_histories= []

            for client in selected:
                client.set_global_weights(deepcopy(global_params))
                updated_params, history, n_samples = client.train()
                client_updates.append((updated_params, n_samples))
                round_histories.append(history)

                last_loss = history["loss"][-1]
                print(f"│  Hospital-{client.id:02d}:  "
                      f"train_loss={last_loss:.4f}")

            # 3. FedAvg aggregation
            self.aggregate(client_updates)

            # 4. Global evaluation
            global_metrics = evaluate_model(
                self.global_model, X_test, y_test, self.device)

            elapsed = time.time() - t0
            print(f"│")
            print(f"│  [Global]  evaluation complete")
            print(f"└── Round {rnd} done in {elapsed:.1f}s\n")

            # Log for visualisation
            avg_client_loss = np.mean([h["loss"][-1] for h in round_histories])
            avg_client_acc  = np.mean([h["accuracy"][-1] for h in round_histories])
            self.round_logs.append({
                "round"      : rnd,
                "avg_client_loss": float(avg_client_loss),
                "avg_client_acc" : float(avg_client_acc),
                **{k: float(v) for k, v in global_metrics.items()
                   if k not in ("confusion_matrix", "probs")}
            })

        print("\n" + "═"*60)
        print("  FEDERATED LEARNING – TRAINING COMPLETE")
        print("═"*60)
        return self.round_logs

    def save_global_model(self, path: str):
        torch.save(self.global_model.state_dict(), path)
        print(f"[Server] Global model saved → {path}")

    def load_global_model(self, path: str):
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(state_dict)
        print(f"[Server] Global model loaded ← {path}")

    def save_logs(self, path: str):
        with open(path, "w") as f:
            json.dump(self.round_logs, f, indent=2)
        print(f"[Server] Logs saved → {path}")
