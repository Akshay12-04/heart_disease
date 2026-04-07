"""
=============================================================
  data_preprocessor.py  –  Load, encode, normalise, partition
=============================================================
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

CATEGORICAL_COLS = [
    "gender", "ecg_result", "smoking_status",
    "physical_activity_level", "sleep_quality",
    "diet_quality", "salt_intake"
]
TARGET_COL = "heart_disease_risk"


def load_and_preprocess(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    print(f"[Preprocessor] Loading: {data_path}")
    df = pd.read_csv(data_path)
    missing_cols = [c for c in CATEGORICAL_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    print(f"[Preprocessor] Shape  : {df.shape}")
    print(f"[Preprocessor] Target :\n{df[TARGET_COL].value_counts()}\n")

    # Fill missing
    df = df.fillna(df.median(numeric_only=True))
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categoricals
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    uniq = np.unique(y)
    if not np.array_equal(uniq, np.array([0, 1])) and not np.array_equal(uniq, np.array([0])) and not np.array_equal(uniq, np.array([1])):
        raise ValueError(f"Target column '{TARGET_COL}' must be binary (0/1). Found: {uniq.tolist()}")
    feature_names = [c for c in df.columns if c != TARGET_COL]

    return X, y, feature_names


def split_into_hospital_partitions(X, y, num_hospitals=5, iid=True, seed=42):
    if num_hospitals < 2:
        raise ValueError("num_hospitals must be >= 2")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if len(X) < num_hospitals * 5:
        raise ValueError("Dataset too small for requested number of hospitals")

    rng = np.random.default_rng(seed)
    n = len(X)
    chunk = n // num_hospitals

    if iid:
        indices = rng.permutation(n)
    else:
        # Non-IID but realistic: each hospital still gets both classes,
        # with varying positive ratios sampled from a beta distribution.
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        total_pos = len(pos_idx)
        total_neg = len(neg_idx)
        target_sizes = [chunk] * num_hospitals
        target_sizes[-1] += n - chunk * num_hospitals
        pos_ratios = rng.beta(a=1.5, b=1.5, size=num_hospitals)
        pos_ratios = np.clip(pos_ratios, 0.10, 0.90)
        raw_pos = np.array([int(r * s) for r, s in zip(pos_ratios, target_sizes)])
        desired_pos = np.clip(raw_pos, 1, np.array(target_sizes) - 1)

        # Balance desired positives to exact global class counts.
        diff = int(desired_pos.sum() - total_pos)
        while diff != 0:
            if diff > 0:
                candidates = np.where(desired_pos > 1)[0]
                if len(candidates) == 0:
                    break
                i = int(rng.choice(candidates))
                desired_pos[i] -= 1
                diff -= 1
            else:
                candidates = np.where(desired_pos < (np.array(target_sizes) - 1))[0]
                if len(candidates) == 0:
                    break
                i = int(rng.choice(candidates))
                desired_pos[i] += 1
                diff += 1
    partitions = []
    for i in range(num_hospitals):
        if iid:
            start = i * chunk
            end = start + chunk if i < num_hospitals - 1 else n
            idx = indices[start:end]
        else:
            pos_take = int(desired_pos[i])
            size = target_sizes[i]
            neg_take = size - pos_take
            pos_slice = pos_idx[:pos_take]
            neg_slice = neg_idx[:neg_take]
            pos_idx = pos_idx[pos_take:]
            neg_idx = neg_idx[neg_take:]
            idx = np.concatenate([pos_slice, neg_slice])
            rng.shuffle(idx)

        X_h, y_h = X[idx], y[idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_h, y_h, test_size=0.2, random_state=seed, stratify=y_h)
        partitions.append({"X_train": X_tr, "y_train": y_tr,
                           "X_val": X_val, "y_val": y_val,
                           "hospital_id": i+1, "size": len(X_h)})
        print(f"  Hospital-{i+1:02d}: {len(X_tr)} train | {len(X_val)} val | "
              f"label_ratio={y_h.mean():.2f}")
    return partitions


def get_global_test_set(X, y, test_size=0.15, seed=42):
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    return X_test, y_test


def split_train_and_global_test(X, y, test_size=0.15, seed=42):
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_scaler_on_train(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return X_train_scaled, X_test_scaled, scaler


def save_scaler(scaler, path):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)

def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)
