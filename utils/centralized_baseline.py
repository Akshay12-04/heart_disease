"""
=============================================================
  centralized_baseline.py
  Trains a centralised model (all data pooled) as a comparison
  baseline against the federated approach.
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from models.heart_model import HeartDiseaseNet, train_local, evaluate_model


def train_centralized(X_train, y_train, X_test, y_test,
                      epochs=20, batch_size=64, lr=1e-3, device="cpu"):
    """
    Train a single model on ALL data pooled together.
    Returns final metrics dict.
    """
    print("\n[Centralized Baseline] Training on full pooled dataset …")
    input_dim = X_train.shape[1]
    model     = HeartDiseaseNet(input_dim).to(device)

    history = train_local(model, X_train, y_train,
                          epochs=epochs, batch_size=batch_size,
                          lr=lr, device=device)

    metrics = evaluate_model(model, X_test, y_test, device)
    print("[Centralized] evaluation complete")
    return metrics, history, model
