"""
=============================================================
  hospital_client.py
  Represents one hospital in the federated-learning system.

  Privacy guarantee
  ─────────────────
  • Each HospitalClient ONLY sends model weight deltas (gradients
    summarised as updated weights) to the server.
  • Raw patient records never leave the hospital boundary.
  • Optional: Differential Privacy (Gaussian noise) on updates.
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import copy

from models.heart_model import HeartDiseaseNet, get_parameters, set_parameters, train_local, evaluate_model


class HospitalClient:
    """
    Simulates one hospital node.

    Responsibilities
    ----------------
    1. Receive global model weights from the server.
    2. Train locally on private patient data.
    3. Return updated weights (+ optional DP noise).
    4. Report local validation metrics.
    """

    def __init__(self, hospital_id: int, X_train, y_train, X_val, y_val,
                 input_dim: int, device: str = "cpu",
                 local_epochs: int = 5, batch_size: int = 32, lr: float = 1e-3,
                 dp_enabled: bool = False, dp_noise_multiplier: float = 0.01):

        self.id          = hospital_id
        self.X_train     = X_train
        self.y_train     = y_train
        self.X_val       = X_val
        self.y_val       = y_val
        self.input_dim   = input_dim
        self.device      = device
        self.local_epochs= local_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.dp_enabled  = dp_enabled
        self.dp_noise_multiplier = dp_noise_multiplier

        # Local model (weights will be set by server before each round)
        self.model = HeartDiseaseNet(input_dim)

    # ── Receive global weights ──────────────────────────────────────────────
    def set_global_weights(self, global_params):
        """Load server-aggregated weights into local model."""
        set_parameters(self.model, global_params)

    # ── Local training ──────────────────────────────────────────────────────
    def train(self):
        """
        Train locally, apply optional DP noise, return updated weights.
        """
        history = train_local(
            self.model, self.X_train, self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device
        )
        updated_params = get_parameters(self.model)

        # ── Differential Privacy (optional) ────────────────────────────────
        if self.dp_enabled:
            updated_params = self._add_dp_noise(updated_params)

        return updated_params, history, len(self.X_train)

    # ── Validation ──────────────────────────────────────────────────────────
    def validate(self):
        metrics = evaluate_model(self.model, self.X_val, self.y_val, self.device)
        return metrics

    # ── Differential Privacy helper ─────────────────────────────────────────
    def _add_dp_noise(self, params):
        """
        Add calibrated Gaussian noise to model updates.

        This is a simplified local DP mechanism:
          noise ~ N(0, σ²I)  where σ = noise_multiplier × sensitivity

        In production use Opacus (PyTorch) or TF Privacy for rigorous DP.
        """
        noisy = []
        for p in params:
            sensitivity = np.linalg.norm(p)  # L2 sensitivity
            noise = np.random.normal(
                0,
                self.dp_noise_multiplier * sensitivity,
                p.shape
            ).astype(np.float32)
            noisy.append(p + noise)
        return noisy

    def __repr__(self):
        return (f"<HospitalClient id={self.id}  "
                f"train={len(self.X_train)}  val={len(self.X_val)}  "
                f"dp={'ON' if self.dp_enabled else 'OFF'}>")
