"""
=============================================================
  heart_model.py  –  Neural-Network model shared by all nodes
=============================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)


class HeartDiseaseNet(nn.Module):
    """Feed-forward NN for binary heart-disease prediction."""
    def __init__(self, input_dim: int):
        super(HeartDiseaseNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def get_parameters(model):
    return [p.data.cpu().numpy() for p in model.parameters()]


def set_parameters(model, parameters):
    for p, new_val in zip(model.parameters(), parameters):
        p.data = torch.tensor(new_val, dtype=torch.float32)


def train_local(model, X_train, y_train,
                epochs=5, batch_size=32, lr=1e-3, device="cpu"):
    model.to(device)
    model.train()
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history = {"loss": [], "accuracy": []}
    for epoch in range(epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_b)
            correct    += ((out >= 0.5).long() == y_b.long()).sum().item()
            total      += len(y_b)
        history["loss"].append(epoch_loss / total)
        history["accuracy"].append(correct / total)
    return history


def evaluate_model(model, X_test, y_test, device="cpu"):
    model.to(device)
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_t).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy" : accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall"   : recall_score(y_test, preds, zero_division=0),
        "f1"       : f1_score(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
        "probs"    : probs
    }
