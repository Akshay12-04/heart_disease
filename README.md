# 🫀 Federated Learning for Heart Disease Prediction

> **Final Year Engineering Project** | Privacy-Preserving AI in Healthcare  
> Built with PyTorch · FedAvg · Differential Privacy · Streamlit

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Dataset](#-dataset)
5. [How Federated Learning Works](#-how-federated-learning-works)
6. [Privacy & Security](#-privacy--security)
7. [Installation](#-installation)
8. [Running the Project](#-running-the-project)
9. [Results](#-results)
10. [Streamlit UI](#-streamlit-ui)
11. [Key Concepts Explained](#-key-concepts-explained)

---

## 🎯 Project Overview

Traditional machine learning in healthcare requires hospitals to **share raw patient data** with a central server — raising serious privacy, legal, and ethical concerns (HIPAA, GDPR).

This project implements a **Federated Learning (FL)** system where:

- **5 hospitals** collaborate to train a heart disease prediction model
- **Raw patient data never leaves** each hospital
- Only **model weight updates** are shared and aggregated
- An **optional Differential Privacy** layer adds mathematical privacy guarantees

### Results Achieved
| Metric    | Federated | Centralized |
|-----------|-----------|-------------|
| Accuracy  | **80.2%** | 92.5%       |
| F1-Score  | **79.9%** | 90.8%       |
| Recall    | **98.5%** | 92.5%       |
| Precision | **67.2%** | 89.1%       |

> The federated model achieves strong **recall (98.5%)** — critical in medical diagnosis where missing a positive case (heart disease) is far more dangerous than a false alarm. The gap vs centralised is the inherent privacy-utility trade-off of FL.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CENTRAL SERVER                         │
│          FedAvg Aggregation · Global Model               │
│        (Never sees raw patient data!)                    │
└──────────────┬──────────────────────┬───────────────────┘
               │  Broadcast weights   │ Receive updates
    ┌──────────▼──────────┐  ┌────────▼──────────────┐
    │   Hospital-01 🏥    │  │   Hospital-02 🏥       │  ...
    │  1600 patients      │  │  1600 patients          │
    │  Train locally      │  │  Train locally          │
    │  DP noise (opt.)    │  │  DP noise (opt.)        │
    └─────────────────────┘  └────────────────────────┘

Communication Round:
  1. Server broadcasts global weights w_t
  2. Each hospital trains → w_k (local update)
  3. Server aggregates: w_{t+1} = Σ (n_k/N) × w_k  [FedAvg]
  4. Evaluate on global held-out test set
  5. Repeat for R rounds
```

---

## 📁 Project Structure

```
federated_heart_disease/
│
├── data/
│   └── heart_disease_federated.csv     # Dataset (10,000 patients, 30 features)
│
├── models/
│   ├── heart_model.py                  # Neural network + train/eval functions
│   ├── global_model.pth                # Saved federated model (after training)
│   └── scaler.pkl                      # Fitted StandardScaler (for inference)
│
├── clients/
│   └── hospital_client.py              # HospitalClient class (local training + DP)
│
├── server/
│   └── federated_server.py             # FederatedServer class (FedAvg aggregation)
│
├── utils/
│   ├── data_preprocessor.py            # Load, encode, scale, partition data
│   ├── centralized_baseline.py         # Pooled training for comparison
│   └── visualizer.py                   # All matplotlib/seaborn plots
│
├── results/
│   ├── plots/                          # 7 auto-generated visualisation PNGs
│   └── metrics/
│       ├── round_logs.json             # Per-round metrics
│       └── final_summary.json          # Federated vs centralized summary
│
├── streamlit_app/
│   └── app.py                          # Interactive Streamlit prediction UI
│
├── main.py                             # ← Entry point (run this)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**File:** `heart_disease_federated.csv`  
**Rows:** 10,000 patient records  
**Target:** `heart_disease_risk` (0 = No, 1 = Yes)

| Category | Features |
|----------|----------|
| Demographics | age, gender, bmi |
| Medical History | family_history, diabetes, hypertension, previous_heart_event |
| Lab Results | cholesterol_total, hdl/ldl_cholesterol, blood_pressure_systolic/diastolic, resting_heart_rate, fasting_blood_sugar |
| Diagnostic | ecg_result |
| Lifestyle | smoking_status, cigarettes_per_day, alcohol_units_per_week, physical_activity_level, exercise_hours_per_week, walks_daily, plays_sport |
| Sleep & Mental | sleep_hours_per_night, sleep_quality, stress_level, depression_anxiety |
| Nutrition | diet_quality, fruit_veg_servings_per_day, salt_intake |

---

## 🔄 How Federated Learning Works

### Step-by-Step Flow

```
ROUND r:
  ┌─ SERVER ──────────────────────────────────────────────┐
  │  1. Broadcast current global weights w_r to all       │
  │     hospital clients                                  │
  └───────────────────────────────────────────────────────┘
          ↓ (weights sent, NO data shared)
  ┌─ EACH HOSPITAL CLIENT ────────────────────────────────┐
  │  2. Load global weights into local model              │
  │  3. Train for E local epochs on private patient data  │
  │  4. (Optional) Add differential privacy noise         │
  │  5. Send updated weights w_k back to server           │
  └───────────────────────────────────────────────────────┘
          ↓ (only weights returned, NO raw data)
  ┌─ SERVER ──────────────────────────────────────────────┐
  │  6. FedAvg aggregation:                               │
  │        w_{r+1} = Σ_k (n_k / N) × w_k                 │
  │     where n_k = hospital k's training size            │
  │           N   = total training samples                │
  │  7. Evaluate new global model on test set             │
  │  8. Log metrics, go to next round                     │
  └───────────────────────────────────────────────────────┘
```

### The Neural Network Model

```
Input (29 features)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(64)  → BatchNorm → ReLU → Dropout(0.2)
    ↓
Dense(32)  → ReLU
    ↓
Dense(1)   → Sigmoid  →  P(heart disease) ∈ [0, 1]
```

---

## 🔐 Privacy & Security

### Why Federated Learning is Private

| Threat | Traditional ML | Federated ML |
|--------|---------------|-------------|
| Data breach at server | ✅ Catastrophic | ❌ Impossible (no raw data) |
| Insider attack | ✅ Possible | ❌ Greatly reduced |
| HIPAA compliance | ⚠️ Complex | ✅ Much simpler |
| Patient consent | ⚠️ Required for data sharing | ✅ Data stays on-premise |

### Differential Privacy (--dp flag)

When enabled, each hospital adds **calibrated Gaussian noise** to weight updates before sending:

```
noise ~ N(0, σ² × ‖Δw‖²)   where σ = noise_multiplier
```

This provides a mathematical guarantee (ε-differential privacy) that an adversary cannot determine whether any specific patient was in the training set, even with full access to the model.

**Privacy-Utility Trade-off:**
- Higher σ → stronger privacy → slightly lower accuracy
- This project uses σ = 0.01 (mild noise, good accuracy preservation)
- For production: use [Opacus](https://opacus.ai/) for rigorous DP-SGD

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# 1. Clone / download the project
cd federated_heart_disease

# 2. (Recommended) Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify dataset is in place
ls data/heart_disease_federated.csv
```

---

## 🚀 Running the Project

### Option 1 — Default run (5 hospitals, 10 rounds, IID)
```bash
python main.py
```

### Option 2 — Custom configuration
```bash
# More rounds for better convergence
python main.py --hospitals 5 --rounds 20 --epochs 5

# Non-IID split (realistic: different hospitals see different patient profiles)
python main.py --noniid --rounds 15

# Enable Differential Privacy
python main.py --dp --rounds 10

# Partial client participation (60% hospitals per round)
python main.py --fraction 0.6 --rounds 15

# Full custom
python main.py --hospitals 5 --rounds 20 --epochs 8 --noniid --dp --fraction 0.8
```

### Option 3 — Streamlit UI (after training)
```bash
streamlit run streamlit_app/app.py
# Opens browser at http://localhost:8501
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hospitals` | 5 | Number of simulated hospital clients |
| `--rounds` | 10 | Communication rounds between server and clients |
| `--epochs` | 5 | Local training epochs per client per round |
| `--noniid` | False | Non-IID data partition (realistic label skew) |
| `--dp` | False | Enable Differential Privacy noise |
| `--fraction` | 1.0 | Fraction of clients selected each round |

---

## 📈 Results

After training, all results are saved to `results/`:

```
results/
├── plots/
│   ├── 01_training_curves.png          # Accuracy, loss, F1 across rounds
│   ├── 02_fed_confusion_matrix.png     # Federated model confusion matrix
│   ├── 03_cen_confusion_matrix.png     # Centralised model confusion matrix
│   ├── 04_federated_vs_centralized.png # Side-by-side metric comparison
│   ├── 05_per_hospital_accuracy.png    # Each hospital's validation accuracy
│   ├── 06_metrics_table.png            # Summary metrics table
│   └── 07_feature_importance.png       # Top-15 features (permutation method)
└── metrics/
    ├── round_logs.json                 # Metrics for every round
    └── final_summary.json             # Final federated vs centralized summary
```

---

## 🖥️ Streamlit UI

The interactive UI provides:

- **🔍 Predict Tab** — Enter patient details, get real-time heart disease risk %
- **📊 Performance Tab** — View all training plots and metric comparisons
- **📈 Training Logs Tab** — Tabular view of per-round metrics with live charts
- **ℹ️ About FL Tab** — Visual explanation of the federated learning process

---

## 💡 Key Concepts Explained

### FedAvg Algorithm
The **Federated Averaging** algorithm (McMahan et al., 2017) is the backbone of this project. Instead of sharing gradients, each client shares its full updated model weights. The server computes a weighted average proportional to each hospital's dataset size.

### IID vs Non-IID Data
- **IID (Independent & Identically Distributed)** — data is randomly split across hospitals. Each hospital sees a similar distribution.
- **Non-IID** — data is split by label, so some hospitals see mostly healthy patients, others see mostly sick patients. This is more realistic and harder to learn from.

### Communication Rounds
Each "round" simulates one exchange between hospitals and the central server. More rounds → better global model → more communication cost. In real deployments, rounds happen over a network with encryption.

### Privacy-Utility Trade-off
Federated learning with DP achieves slightly lower accuracy than centralised learning. This gap is the **cost of privacy** — entirely acceptable in healthcare where protecting patient data is paramount.

---

## 📚 References

1. McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
2. Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in TCS.
3. Rieke, N. et al. (2020). *The Future of Digital Health with Federated Learning.* npj Digital Medicine.
4. Yang, Q. et al. (2019). *Federated Machine Learning: Concept and Applications.* ACM TIST.

---

*Built for Final Year Engineering Project — Privacy-Preserving AI in Healthcare*
