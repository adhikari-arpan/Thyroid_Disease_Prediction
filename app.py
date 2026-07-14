"""
Thyroid Disease Prediction — Streamlit Demo App
NCIT ML Group Case Study

This app loads YOUR team's actual trained models directly from the project
structure — nothing here is retrained or faked.

EXPECTED FOLDER STRUCTURE (place app.py in the project ROOT, next to the
data/ and models/ folders shown in your file tree):

Thyroid_Disease_Prediction/
├── app.py                          <-- this file goes here
├── data/
│   └── processed/
│       ├── class_labels.json
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── random_forest_baseline.joblib
│   ├── baseline_thyroid_ann.keras
│   ├── tuned_thyroid_ann.keras
│   └── thyroid_ann_metadata.json

HOW TO RUN:
1. pip install streamlit scikit-learn joblib pandas numpy tensorflow
2. From the project root: streamlit run app.py
3. Opens automatically at http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Thyroid Disease Predictor",
    page_icon="🩺",
    layout="wide",
)

DATA_DIR = "data/processed"
MODELS_DIR = "models"

# ============================================================
# LOAD EVERYTHING (cached so it only runs once per session)
# ============================================================
@st.cache_resource
def load_everything():
    errors = []

    # --- Feature order: read directly from train.csv so it always matches
    # exactly what the models were actually trained on ---
    train_path = os.path.join(DATA_DIR, "train.csv")
    try:
        train_df = pd.read_csv(train_path)
        feature_order = [c for c in train_df.columns if c != "class"]
    except Exception as e:
        errors.append(f"Could not read {train_path}: {e}")
        feature_order = None

    # --- Class labels (readable names) ---
    labels_path = os.path.join(DATA_DIR, "class_labels.json")
    try:
        with open(labels_path) as f:
            raw_labels = json.load(f)
        class_labels = {int(k): v for k, v in raw_labels.items()}
    except Exception as e:
        errors.append(f"Could not read {labels_path}: {e}")
        class_labels = {1: "hyperthyroid", 2: "hypothyroid", 3: "normal"}

    # --- Random Forest (baseline + tuned) ---
    rf_baseline_path = os.path.join(MODELS_DIR, "random_forest_baseline.joblib")
    rf_tuned_path = os.path.join(MODELS_DIR, "random_forest_tuned.joblib")
    try:
        rf_baseline_model = joblib.load(rf_baseline_path)
    except Exception as e:
        errors.append(f"Could not load {rf_baseline_path}: {e}")
        rf_baseline_model = None
    try:
        rf_tuned_model = joblib.load(rf_tuned_path)
    except Exception as e:
        errors.append(f"Could not load {rf_tuned_path}: {e}")
        rf_tuned_model = None

    # --- ANN models (Keras) ---
    ann_baseline, ann_tuned = None, None
    try:
        from tensorflow import keras
        ann_baseline_path = os.path.join(MODELS_DIR, "baseline_thyroid_ann.keras")
        ann_tuned_path = os.path.join(MODELS_DIR, "tuned_thyroid_ann.keras")
        ann_baseline = keras.models.load_model(ann_baseline_path)
        ann_tuned = keras.models.load_model(ann_tuned_path)
    except Exception as e:
        errors.append(f"Could not load ANN models (is tensorflow installed?): {e}")

    # --- Metadata (optional, just for display) ---
    metadata = None
    meta_path = os.path.join(MODELS_DIR, "thyroid_ann_metadata.json")
    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except Exception:
        pass  # optional, don't hard-fail if missing

    return (feature_order, class_labels, rf_baseline_model, rf_tuned_model,
            ann_baseline, ann_tuned, metadata, errors)


(feature_order, class_labels, rf_baseline_model, rf_tuned_model,
 ann_baseline, ann_tuned, metadata, load_errors) = load_everything()

CLASS_COLORS = {1: "#e67e22", 2: "#3498db", 3: "#2ecc71"}

# ============================================================
# HEADER
# ============================================================
st.title("🩺 Thyroid Disease Predictor")
st.caption(
    "NCIT Machine Learning Group Case Study — live comparison of Random Forest "
    "and ANN, both baseline & tuned, using our actual trained models."
)

if load_errors:
    st.error(
        "Some files couldn't be loaded. Make sure app.py is in the project "
        "ROOT folder (same level as data/ and models/), and that all "
        "dependencies are installed. Details below:"
    )
    for e in load_errors:
        st.code(e)
    st.stop()

st.divider()

# ============================================================
# SIDEBAR — model info
# ============================================================
with st.sidebar:
    st.header("About these models")
    st.write(
        "Three models are compared side by side, all trained on the same "
        "UCI ANN-Thyroid dataset:"
    )
    st.markdown("- **Random Forest** (baseline)")
    st.markdown("- **Random Forest** (tuned)")
    st.markdown("- **ANN** (baseline)")
    st.markdown("- **ANN** (tuned via Keras Tuner)")
    st.divider()
    st.markdown("**Classes predicted:**")
    for k in sorted(class_labels.keys()):
        st.markdown(f"- Class {k}: {class_labels[k]}")
    if metadata:
        st.divider()
        st.markdown("**Tuned ANN — best hyperparameters found:**")
        st.json(metadata.get("best_hyperparameters", metadata), expanded=False)
    st.divider()
    st.markdown(
        "**Note:** Academic demo only — not a real medical diagnostic tool."
    )

# ============================================================
# INPUT FORM
# ============================================================
st.subheader("Enter Patient Information")
st.caption(
    "Values below are in normalized form (0–1), matching how the training "
    "data itself is scaled in this dataset. Use the reference hints as a guide."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    age = st.slider("Age (normalized, 0=youngest, 1=oldest)", 0.0, 1.0, 0.45, 0.01)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)

    st.markdown("**Medication / Treatment History**")
    on_thyroxine = st.checkbox("Currently on thyroxine medication")
    query_on_thyroxine = st.checkbox("Uncertain if on thyroxine")
    on_antithyroid_meds = st.checkbox("On anti-thyroid medication")
    thyroid_surgery = st.checkbox("History of thyroid surgery")
    i131_treatment = st.checkbox("History of radioactive iodine (I131) treatment")
    lithium = st.checkbox("Currently on lithium")

with col2:
    st.markdown("**Clinical Suspicion / Risk Factors**")
    query_hypothyroid = st.checkbox("Doctor suspects hypothyroidism")
    query_hyperthyroid = st.checkbox("Doctor suspects hyperthyroidism")
    sick = st.checkbox("Currently sick / acutely ill")
    pregnant = st.checkbox("Currently pregnant")
    goitre = st.checkbox("Goitre present (enlarged thyroid)")
    tumor = st.checkbox("Thyroid tumor present")
    hypopituitary = st.checkbox("Hypopituitarism present")
    psych = st.checkbox("Psychiatric condition present")

with col3:
    st.markdown("**Lab Values (normalized, dataset scale)**")
    tsh = st.number_input("TSH", min_value=0.0, max_value=0.6, value=0.005, step=0.001, format="%.4f",
                           help="Higher = possible underactive thyroid. Lower = possible overactive thyroid.")
    t3 = st.number_input("T3", min_value=0.0, max_value=0.2, value=0.020, step=0.001, format="%.4f")
    tt4 = st.number_input("TT4", min_value=0.0, max_value=0.7, value=0.110, step=0.001, format="%.4f")
    t4u = st.number_input("T4U", min_value=0.0, max_value=0.3, value=0.098, step=0.001, format="%.4f")
    fti = st.number_input("FTI", min_value=0.0, max_value=0.7, value=0.113, step=0.001, format="%.4f")

st.divider()

# ============================================================
# PREDICT
# ============================================================
if st.button("🔍 Predict Thyroid Status", type="primary", use_container_width=True):

    raw_input = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "on_thyroxine": int(on_thyroxine),
        "query_on_thyroxine": int(query_on_thyroxine),
        "on_antithyroid_meds": int(on_antithyroid_meds),
        "sick": int(sick),
        "pregnant": int(pregnant),
        "thyroid_surgery": int(thyroid_surgery),
        "I131_treatment": int(i131_treatment),
        "query_hypothyroid": int(query_hypothyroid),
        "query_hyperthyroid": int(query_hyperthyroid),
        "lithium": int(lithium),
        "goitre": int(goitre),
        "tumor": int(tumor),
        "hypopituitary": int(hypopituitary),
        "psych": int(psych),
        "TSH": tsh,
        "T3": t3,
        "TT4": tt4,
        "T4U": t4u,
        "FTI": fti,
    }

    # Build input in the EXACT column order the models were trained on
    input_df = pd.DataFrame([raw_input])[feature_order]
    input_array = input_df.values.astype("float32")

    st.subheader("Prediction Results — Side by Side")

    rf_base_col, rf_tuned_col, ann_base_col, ann_tuned_col = st.columns(4)

    def render_result(container, title, predicted_class, confidence, all_probs, class_order):
        color = CLASS_COLORS.get(predicted_class, "#333333")
        with container:
            st.markdown(f"**{title}**")
            st.markdown(
                f"""
                <div style="padding:14px;border-radius:8px;background-color:{color}22;
                            border:2px solid {color};">
                    <p style="margin:0;font-weight:bold;color:{color};">
                        {class_labels[predicted_class]}
                    </p>
                    <p style="margin:4px 0 0 0;">Confidence: <b>{confidence*100:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prob_df = pd.DataFrame({
                "Class": [class_labels[c] for c in class_order],
                "Probability": all_probs,
            }).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Class"), height=180)

    # --- Random Forest baseline ---
    rf_base_pred = rf_baseline_model.predict(input_df)[0]
    rf_base_proba = rf_baseline_model.predict_proba(input_df)[0]
    rf_base_classes = rf_baseline_model.classes_
    render_result(rf_base_col, "Random Forest (baseline)", rf_base_pred,
                   rf_base_proba[list(rf_base_classes).index(rf_base_pred)], rf_base_proba, rf_base_classes)

    # --- Random Forest tuned ---
    rf_tuned_pred = rf_tuned_model.predict(input_df)[0]
    rf_tuned_proba = rf_tuned_model.predict_proba(input_df)[0]
    rf_tuned_classes = rf_tuned_model.classes_
    render_result(rf_tuned_col, "Random Forest (tuned)", rf_tuned_pred,
                   rf_tuned_proba[list(rf_tuned_classes).index(rf_tuned_pred)], rf_tuned_proba, rf_tuned_classes)

    # --- ANN baseline (Keras outputs 0-indexed classes -> shift back to 1/2/3) ---
    ann_base_proba = ann_baseline.predict(input_array, verbose=0)[0]
    ann_base_pred_idx = int(np.argmax(ann_base_proba))
    ann_base_pred = ann_base_pred_idx + 1
    ann_classes_display = [i + 1 for i in range(len(ann_base_proba))]
    render_result(ann_base_col, "ANN (baseline)", ann_base_pred,
                   ann_base_proba[ann_base_pred_idx], ann_base_proba, ann_classes_display)

    # --- ANN tuned ---
    ann_tuned_proba = ann_tuned.predict(input_array, verbose=0)[0]
    ann_tuned_pred_idx = int(np.argmax(ann_tuned_proba))
    ann_tuned_pred = ann_tuned_pred_idx + 1
    render_result(ann_tuned_col, "ANN (tuned)", ann_tuned_pred,
                   ann_tuned_proba[ann_tuned_pred_idx], ann_tuned_proba, ann_classes_display)

    st.divider()

    # Agreement check — nice talking point for the defense
    preds = {
        "Random Forest (baseline)": rf_base_pred,
        "Random Forest (tuned)": rf_tuned_pred,
        "ANN (baseline)": ann_base_pred,
        "ANN (tuned)": ann_tuned_pred,
    }
    unique_preds = set(preds.values())
    if len(unique_preds) == 1:
        st.success("✅ All three models agree on this prediction.")
    else:
        st.warning(
            "⚠️ The models disagree on this patient — a good discussion point for "
            "your defense about where each model's strengths/weaknesses show up."
        )
        st.write(preds)

    st.info(
        "This prediction reflects patterns learned from historical patient data. "
        "It is intended for academic demonstration only and is not a substitute "
        "for professional medical evaluation."
    )

st.divider()
st.caption("Built for the NCIT BE Computer Engineering — Machine Learning Group Case Study")