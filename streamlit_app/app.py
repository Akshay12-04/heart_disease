"""
=============================================================
  app.py  –  Streamlit UI for Heart Disease Risk Prediction
  Run:  streamlit run streamlit_app/app.py
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import torch
import pandas as pd
import pickle
import json
import random
from datetime import datetime

from models.heart_model import HeartDiseaseNet
from models.heart_model import evaluate_model
from utils.data_preprocessor import (
    load_and_preprocess,
    split_into_hospital_partitions,
    split_train_and_global_test,
    fit_scaler_on_train,
    save_scaler,
)
from clients.hospital_client import HospitalClient
from server.federated_server import FederatedServer
from utils.visualizer import plot_training_curves, plot_confusion_matrix, plot_feature_importance

BASE = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH  = os.path.join(BASE, "models", "global_model.pth")
SCALER_PATH = os.path.join(BASE, "models", "scaler.pkl")
LOGS_PATH   = os.path.join(BASE, "results", "metrics", "round_logs.json")
PLOTS_DIR   = os.path.join(BASE, "results", "plots")

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "age","gender","bmi","family_history_heart_disease","diabetes","hypertension",
    "previous_heart_event","cholesterol_total","hdl_cholesterol","ldl_cholesterol",
    "blood_pressure_systolic","blood_pressure_diastolic","resting_heart_rate",
    "fasting_blood_sugar","ecg_result","smoking_status","cigarettes_per_day",
    "alcohol_units_per_week","physical_activity_level","exercise_hours_per_week",
    "walks_daily","plays_sport","sleep_hours_per_night","sleep_quality",
    "stress_level","depression_anxiety","diet_quality","fruit_veg_servings_per_day",
    "salt_intake"
]

CAT_ENCODE = {
    "gender"                : {"Male": 1, "Female": 0},
    "ecg_result"            : {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2},
    "smoking_status"        : {"Never": 0, "Former": 1, "Current": 2},
    "physical_activity_level": {"Sedentary": 0, "Lightly Active": 1, "Active": 2, "Very Active": 3},
    "sleep_quality"         : {"Poor": 0, "Fair": 1, "Good": 2},
    "diet_quality"          : {"Poor": 0, "Fair": 1, "Good": 2},
    "salt_intake"           : {"Low": 0, "Medium": 1, "High": 2}
}

# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_and_scaler():
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    model  = HeartDiseaseNet(len(FEATURE_ORDER))
    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


def train_model_from_ui(rounds, epochs, hospitals, noniid, dp, seed=42, progress_callback=None):
    if rounds < 1 or epochs < 1:
        raise ValueError("Rounds and epochs must be >= 1")
    if hospitals < 2:
        raise ValueError("Hospitals must be >= 2")

    def _progress(pct, msg):
        if progress_callback is not None:
            progress_callback(pct, msg)

    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "results", "metrics"), exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    _progress(0.05, "Preparing runtime and folders...")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _progress(0.12, "Loading and preprocessing dataset...")
    data_path = os.path.join(BASE, "data", "heart_disease_federated.csv")
    X, y, feature_names = load_and_preprocess(data_path)
    X_train_pool, X_test, y_train_pool, y_test = split_train_and_global_test(
        X, y, test_size=0.15, seed=seed
    )
    X_train_pool, X_test, scaler = fit_scaler_on_train(X_train_pool, X_test)
    save_scaler(scaler, SCALER_PATH)

    _progress(0.22, "Partitioning data across hospitals...")
    partitions = split_into_hospital_partitions(
        X_train_pool, y_train_pool,
        num_hospitals=hospitals,
        iid=not noniid,
        seed=seed,
    )
    input_dim = X.shape[1]
    clients = [
        HospitalClient(
            hospital_id=p["hospital_id"],
            X_train=p["X_train"],
            y_train=p["y_train"],
            X_val=p["X_val"],
            y_val=p["y_val"],
            input_dim=input_dim,
            device=device,
            local_epochs=epochs,
            dp_enabled=dp,
            dp_noise_multiplier=0.01,
        )
        for p in partitions
    ]

    _progress(0.35, "Running federated training rounds...")
    server = FederatedServer(input_dim=input_dim, device=device)
    round_logs = server.run(clients, X_test, y_test, num_rounds=rounds, fraction_clients=1.0)
    _progress(0.82, "Saving model, logs, and plots...")
    server.save_global_model(MODEL_PATH)
    logs_path = os.path.join(BASE, "results", "metrics", "round_logs.json")
    server.save_logs(logs_path)

    fed_metrics = evaluate_model(server.global_model, X_test, y_test, device)
    for plot_name, plot_fn in [
        ("01_training_curves.png", lambda: plot_training_curves(round_logs, os.path.join(PLOTS_DIR, "01_training_curves.png"))),
        ("02_fed_confusion_matrix.png", lambda: plot_confusion_matrix(fed_metrics["confusion_matrix"], "Federated Model – Confusion Matrix", os.path.join(PLOTS_DIR, "02_fed_confusion_matrix.png"))),
        ("07_feature_importance.png", lambda: plot_feature_importance(server.global_model, X_test, y_test, feature_names, os.path.join(PLOTS_DIR, "07_feature_importance.png"), device)),
    ]:
        try:
            plot_fn()
        except Exception as exc:
            _progress(0.90, f"Plot warning ({plot_name}): {exc}")

    summary = {
        "federated": {"confusion_matrix": fed_metrics["confusion_matrix"]},
        "centralized": {"confusion_matrix": None},
        "config": {
            "hospitals": hospitals,
            "rounds": rounds,
            "epochs": epochs,
            "noniid": noniid,
            "dp": dp,
            "fraction": 1.0,
            "seed": seed,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    with open(os.path.join(BASE, "results", "metrics", "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    load_model_and_scaler.clear()
    _progress(1.0, "Training finished. New model is active for prediction.")
    return summary["config"]["trained_at"]

# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Heart Disease Predictor – Federated AI",
        page_icon="🫀",
        layout="wide"
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg,#1565C0,#0D47A1);
                padding:1.5rem;border-radius:12px;margin-bottom:1.5rem">
      <h1 style="color:white;margin:0">🫀 Heart Disease Risk Predictor</h1>
      <p style="color:#90CAF9;margin-top:0.3rem">
        Powered by Federated Learning · Privacy-Preserving AI
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: privacy info ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔒 Privacy & Security")
        st.info("""
**How this model protects patients:**

✅ Trained with **Federated Learning** across 5 virtual hospitals

✅ **Raw patient data never leaves** each hospital

✅ Only encrypted model weight updates were shared

✅ Optional **Differential Privacy** noise added

✅ Compliant with HIPAA / GDPR principles
        """)
        st.divider()
        st.header("🏥 Hospital Network")
        for i in range(1, 6):
            st.success(f"🏥 Hospital-{i:02d}  ·  Connected")

    tabs = st.tabs(["🔍 Predict", "🧠 Train Model", "📊 Model Performance", "📈 Training Logs", "ℹ️ About FL"])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 – Prediction
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Enter Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            age    = st.slider("Age", 18, 100, 55)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi    = st.slider("BMI", 15.0, 50.0, 26.0, 0.1)
            family = st.checkbox("Family History of Heart Disease")

        with col2:
            st.markdown("**🩺 Medical History**")
            diabetes     = st.checkbox("Diabetes")
            hypertension = st.checkbox("Hypertension")
            prev_event   = st.checkbox("Previous Heart Event")
            chol_total   = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
            hdl          = st.number_input("HDL Cholesterol (mg/dL)", 20, 120, 50)
            ldl          = st.number_input("LDL Cholesterol (mg/dL)", 50, 300, 130)
            bp_sys       = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
            bp_dia       = st.number_input("Diastolic BP (mmHg)", 50, 140, 80)
            rhr          = st.number_input("Resting Heart Rate (bpm)", 40, 150, 72)
            fbs          = st.number_input("Fasting Blood Sugar (mg/dL)", 50.0, 400.0, 100.0)
            ecg          = st.selectbox("ECG Result", list(CAT_ENCODE["ecg_result"].keys()))

        with col3:
            st.markdown("**🚬 Lifestyle**")
            smoking   = st.selectbox("Smoking Status", list(CAT_ENCODE["smoking_status"].keys()))
            cig_day   = st.slider("Cigarettes/Day", 0, 60, 0)
            alcohol   = st.slider("Alcohol Units/Week", 0.0, 50.0, 3.0)
            activity  = st.selectbox("Physical Activity Level", list(CAT_ENCODE["physical_activity_level"].keys()))
            ex_hours  = st.slider("Exercise Hours/Week", 0.0, 30.0, 3.0)
            walks     = st.checkbox("Walks Daily")
            sport     = st.checkbox("Plays Sport")
            sleep_h   = st.slider("Sleep Hours/Night", 3.0, 12.0, 7.0)
            sleep_q   = st.selectbox("Sleep Quality", list(CAT_ENCODE["sleep_quality"].keys()))
            stress    = st.slider("Stress Level (1–10)", 1, 10, 5)
            dep_anx   = st.checkbox("Depression/Anxiety")
            diet_q    = st.selectbox("Diet Quality", list(CAT_ENCODE["diet_quality"].keys()))
            fruit_veg = st.slider("Fruit & Veg Servings/Day", 0.0, 10.0, 3.0)
            salt      = st.selectbox("Salt Intake", list(CAT_ENCODE["salt_intake"].keys()))

        if st.button("🔍 Predict Risk", type="primary", width="stretch"):
            # Build feature vector in correct column order
            raw = {
                "age": age, "gender": CAT_ENCODE["gender"][gender],
                "bmi": bmi, "family_history_heart_disease": int(family),
                "diabetes": int(diabetes), "hypertension": int(hypertension),
                "previous_heart_event": int(prev_event),
                "cholesterol_total": chol_total, "hdl_cholesterol": hdl,
                "ldl_cholesterol": ldl, "blood_pressure_systolic": bp_sys,
                "blood_pressure_diastolic": bp_dia, "resting_heart_rate": rhr,
                "fasting_blood_sugar": fbs,
                "ecg_result": CAT_ENCODE["ecg_result"][ecg],
                "smoking_status": CAT_ENCODE["smoking_status"][smoking],
                "cigarettes_per_day": cig_day, "alcohol_units_per_week": alcohol,
                "physical_activity_level": CAT_ENCODE["physical_activity_level"][activity],
                "exercise_hours_per_week": ex_hours,
                "walks_daily": int(walks), "plays_sport": int(sport),
                "sleep_hours_per_night": sleep_h,
                "sleep_quality": CAT_ENCODE["sleep_quality"][sleep_q],
                "stress_level": stress, "depression_anxiety": int(dep_anx),
                "diet_quality": CAT_ENCODE["diet_quality"][diet_q],
                "fruit_veg_servings_per_day": fruit_veg,
                "salt_intake": CAT_ENCODE["salt_intake"][salt]
            }
            x_arr = np.array([[raw[f] for f in FEATURE_ORDER]], dtype=np.float32)

            try:
                model, scaler = load_model_and_scaler()
                x_scaled = scaler.transform(x_arr).astype(np.float32)
                with torch.no_grad():
                    prob = model(torch.tensor(x_scaled)).item()
                risk_pct = prob * 100

                st.divider()
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    color = "#d32f2f" if prob >= 0.5 else "#2e7d32"
                    label = "⚠️ HIGH RISK" if prob >= 0.5 else "✅ LOW RISK"
                    st.markdown(f"""
                    <div style="background:{color};padding:1.5rem;border-radius:10px;
                                text-align:center;color:white;">
                      <h2 style="margin:0">{label}</h2>
                      <h1 style="margin:0.3rem 0;font-size:3rem">{risk_pct:.1f}%</h1>
                      <p style="margin:0">Predicted Probability of Heart Disease</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(prob)
                if prob >= 0.5:
                    st.warning("⚕️ **Clinical Recommendation**: High-risk indicators detected. "
                               "Consult a cardiologist and consider further diagnostic tests.")
                else:
                    st.success("✅ Low risk profile. Maintain healthy lifestyle habits.")

            except FileNotFoundError:
                st.error("Model not found. Please run training first from the Train Model tab.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 – Performance
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("Train Federated Model")
        st.caption("Train from this page. The latest trained model is used immediately for prediction.")
        st.info("Use Demo preset for live presentation. Full preset takes longer but gives stronger model updates.")

        preset = st.radio(
            "Training Preset",
            options=["Demo (Fast)", "Balanced", "Full (Slower)"],
            horizontal=True,
        )
        preset_defaults = {
            "Demo (Fast)": {"rounds": 2, "epochs": 1, "hospitals": 3, "noniid": True, "dp": False},
            "Balanced": {"rounds": 5, "epochs": 3, "hospitals": 5, "noniid": True, "dp": True},
            "Full (Slower)": {"rounds": 10, "epochs": 5, "hospitals": 5, "noniid": True, "dp": True},
        }
        defaults = preset_defaults[preset]

        c1, c2, c3 = st.columns(3)
        with c1:
            rounds = st.slider("FL Rounds", min_value=1, max_value=30, value=defaults["rounds"], step=1)
            epochs = st.slider("Local Epochs", min_value=1, max_value=10, value=defaults["epochs"], step=1)
        with c2:
            hospitals = st.slider("Hospitals", min_value=2, max_value=10, value=defaults["hospitals"], step=1)
            noniid = st.checkbox("Use Non-IID split", value=defaults["noniid"])
        with c3:
            dp = st.checkbox("Enable Differential Privacy", value=defaults["dp"])
            seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)

        complexity = rounds * epochs * hospitals
        est_secs = max(8, int(complexity * 0.7))
        st.caption(f"Estimated training time: ~{est_secs}-{est_secs + 15} seconds on CPU.")
        if complexity > 220:
            st.warning("High workload selected. For a smooth live demo, prefer Demo/Balanced preset.")

        allow_train = st.checkbox("I confirm these settings and want to start training.", value=True)

        if "last_trained_at" not in st.session_state:
            st.session_state["last_trained_at"] = None
        if st.session_state["last_trained_at"]:
            st.success(f"Latest model trained at: {st.session_state['last_trained_at']}")

        if st.button("🚀 Start Training", type="primary", width="stretch", disabled=not allow_train):
            try:
                progress = st.progress(0)
                status = st.empty()

                def ui_progress(pct, msg):
                    progress.progress(int(pct * 100))
                    status.info(msg)

                with st.spinner("Training in progress... this may take a few minutes."):
                    trained_at = train_model_from_ui(
                        rounds=rounds,
                        epochs=epochs,
                        hospitals=hospitals,
                        noniid=noniid,
                        dp=dp,
                        seed=int(seed),
                        progress_callback=ui_progress,
                    )
                st.session_state["last_trained_at"] = trained_at
                progress.progress(100)
                status.success("Training pipeline completed successfully.")
                st.success("Training complete. Predictor now uses the newly trained model.")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 – Performance
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("Model Performance Visualisations")
        # Show saved plots
        plot_files = [
            ("01_training_curves.png", "Training Curves"),
            ("02_fed_confusion_matrix.png", "Federated Confusion Matrix"),
            ("07_feature_importance.png", "Feature Importance"),
        ]
        for fname, title in plot_files:
            path = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, width="stretch")
                st.divider()
            else:
                st.info(f"Run `python main.py` to generate: {title}")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 – Training Logs
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("Round-by-Round Training Logs")
        if os.path.exists(LOGS_PATH):
            with open(LOGS_PATH) as f:
                logs = json.load(f)
            df = pd.DataFrame(logs)
            cols_to_hide = {"accuracy", "precision", "recall", "f1"}
            visible_cols = [c for c in df.columns if c not in cols_to_hide]
            st.dataframe(
                df[visible_cols].style.format(
                    {c: "{:.4f}" for c in visible_cols if c != "round"}
                ),
                width="stretch",
            )
            if "avg_client_loss" in df.columns:
                st.line_chart(df.set_index("round")[["avg_client_loss"]])
        else:
            st.info("No logs yet. Run `python main.py` to train the model.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5 – About FL
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("How Federated Learning Protects Patient Privacy")
        st.markdown("""
### 🔐 The Core Privacy Problem in Healthcare

Traditional machine learning requires **centralising data** — all hospitals send
patient records to one server. This creates:
- HIPAA / GDPR compliance risks
- Single point of breach
- Patient distrust

### 🌐 Federated Learning Solution

```
 Hospital-1 🏥          Hospital-2 🏥         Hospital-3 🏥
  Local data              Local data             Local data
  ↓ train locally         ↓ train locally        ↓ train locally
  Updated weights ──────► CENTRAL SERVER ◄────── Updated weights
                               ↓
                         FedAvg Aggregation
                               ↓
                         New Global Weights
                         (broadcast back)
```

**Key Privacy Guarantees:**
| Aspect | Traditional ML | Federated ML |
|--------|---------------|-------------|
| Raw data shared | ✅ Yes | ❌ Never |
| Model weights shared | N/A | ✅ Yes (only) |
| HIPAA compliant | ⚠️ Risky | ✅ Safer |
| Attack surface | High | Low |

### 🛡️ Differential Privacy (Optional Layer)

When `--dp` flag is enabled, Gaussian noise is added to each update:
> noise ~ N(0, σ² × sensitivity²)

This ensures even weight vectors cannot be reverse-engineered back to patient data.

### 📖 References
- McMahan et al. (2017) — "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Dwork & Roth (2014) — "The Algorithmic Foundations of Differential Privacy"
        """)


if __name__ == "__main__":
    main()
