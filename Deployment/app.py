"""
app.py — Streamlit deployment for the ASD Prediction project.

Loads the three tuned models (Decision Tree, Random Forest, XGBoost) and the
saved LabelEncoders produced by train_models.py, and lets the user get a
prediction from one model or compare all three.

This app does NOT retrain anything, does NOT fit new encoders, and does NOT
apply SMOTE to user input (SMOTE is a training-time-only technique). It only
loads what train_models.py already saved and reuses it as-is.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ENCODERS_PATH = BASE_DIR / "encoders.pkl"

# Exact feature order used during training — must never change.
FEATURE_ORDER = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "age", "gender", "ethnicity", "jaundice", "austim",
    "contry_of_res", "used_app_before", "result", "relation",
]

CATEGORICAL_COLUMNS = [
    "gender", "ethnicity", "jaundice", "austim",
    "contry_of_res", "used_app_before", "relation",
]

MODEL_FILES = {
    "Decision Tree": MODELS_DIR / "decision_tree.pkl",
    "Random Forest": MODELS_DIR / "random_forest.pkl",
    "XGBoost": MODELS_DIR / "xgboost.pkl",
}

# Official AQ-10 (Adult) scoring key — Allison, Auyeung & Baron-Cohen (2012),
# University of Cambridge Autism Research Centre. Items 1, 7, 8, 10 score a
# trait-point on AGREE; items 2, 3, 4, 5, 6, 9 are reverse-scored and score a
# trait-point on DISAGREE. This is the same key used to produce the
# A1_Score..A10_Score columns in the original dataset, so it must be applied
# identically here or the collected answers won't mean what the model expects.
AGREE_SCORES_1 = {1, 7, 8, 10}
DISAGREE_SCORES_1 = {2, 3, 4, 5, 6, 9}

AQ_ITEMS = {
    1: "I often notice small sounds when others do not.",
    2: "I usually concentrate more on the whole picture, rather than the small details.",
    3: "I find it easy to do more than one thing at once.",
    4: "If there is an interruption, I can switch back to what I was doing very quickly.",
    5: "I find it easy to 'read between the lines' when someone is talking to me.",
    6: "I know how to tell if someone listening to me is getting bored.",
    7: "When I'm reading a story, I find it difficult to work out the characters' intentions.",
    8: "I like to collect information about categories of things (e.g. types of car, bird, train, plant, etc).",
    9: "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    10: "I find it difficult to work out people's intentions.",
}

st.set_page_config(
    page_title="Autism Spectrum Disorder Prediction",
    page_icon="🧩",
    layout="centered",
)


# ---------------------------------------------------------------------
# Cached loaders — read the saved artifacts once per session
# ---------------------------------------------------------------------
@st.cache_resource
def load_encoders():
    with open(ENCODERS_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_models():
    loaded = {}
    for name, path in MODEL_FILES.items():
        with open(path, "rb") as f:
            loaded[name] = pickle.load(f)
    return loaded


try:
    encoders = load_encoders()
    models = load_models()
    load_error = None
except FileNotFoundError as e:
    encoders, models, load_error = None, None, str(e)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Autism Spectrum Disorder Prediction")
st.caption(
    "A machine learning demo based on the AQ-10 (Adult) screening tool. "
    "This is a class project, **not** a medical diagnosis — please consult "
    "a qualified professional for an actual assessment."
)

if load_error:
    st.error(
        "Model files or `encoders.pkl` were not found.\n\n"
        "Run `python train_models.py` first (with `train.csv` present) to "
        "generate `encoders.pkl` and the files inside `models/`, then "
        "restart this app."
    )
    st.stop()

# ---------------------------------------------------------------------
# AQ-10 screening questions (A1–A10)
# ---------------------------------------------------------------------
st.subheader("AQ-10 Screening Questions")
st.caption(
    "For each statement, say whether you agree or disagree. This uses the "
    "official AQ-10 scoring key, where some items score a point on Agree and "
    "others score a point on Disagree — the same key used to build the "
    "original dataset."
)

aq_answers = {}
cols = st.columns(2)
for i, statement in AQ_ITEMS.items():
    key = f"A{i}_Score"
    with cols[(i - 1) % 2]:
        choice = st.radio(
            f"A{i}: {statement}",
            options=["Agree", "Disagree"],
            horizontal=True,
            key=key,
        )
        if i in AGREE_SCORES_1:
            score = 1 if choice == "Agree" else 0
        else:  # reverse-scored item
            score = 1 if choice == "Disagree" else 0
        aq_answers[key] = score

suggested_result = float(sum(aq_answers.values()))

st.divider()

# ---------------------------------------------------------------------
# Personal & background info
# ---------------------------------------------------------------------
st.subheader("Personal & Background Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)
    gender = st.selectbox("Gender", list(encoders["gender"].classes_))
    ethnicity = st.selectbox("Ethnicity", list(encoders["ethnicity"].classes_))
    jaundice = st.selectbox("Born with jaundice?", list(encoders["jaundice"].classes_))

with col2:
    austim = st.selectbox("Family member with autism?", list(encoders["austim"].classes_))
    contry_of_res = st.selectbox("Country of residence", list(encoders["contry_of_res"].classes_))
    used_app_before = st.selectbox("Used a screening app before?", list(encoders["used_app_before"].classes_))
    relation = st.selectbox("Who is completing this test?", list(encoders["relation"].classes_))

# result = st.number_input(
#     "Result (AQ-10 screening score)",
#     value=suggested_result,
#     step=0.1,
#     format="%.4f",
#     help=(
#         "Pre-filled with the raw AQ-10 total (0-10) computed from your "
#         "answers above using the official scoring key. This is only an "
#         "estimate — in the original dataset this value is not exactly "
#         "reconstructable from the ten answers alone (it varies even between "
#         "records with identical answers), so if you have the real screening "
#         "score from an actual assessment, enter that instead."
#     ),
# )

st.divider()

# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------
st.subheader("Model")
model_choice = st.selectbox(
    "Choose a model",
    ["Decision Tree", "Random Forest", "XGBoost", "Compare All Models"],
)

with st.expander("See which factors each model relies on most (feature importance)"):
    st.caption(
        "Tree-based models don't have to weigh the 19 features equally — "
        "each one learns its own splits from the training data. This shows "
        "what each model actually learned to rely on."
    )
    importance_targets = (
        ["Decision Tree", "Random Forest", "XGBoost"]
        if model_choice == "Compare All Models"
        else [model_choice]
    )
    for name in importance_targets:
        model_obj = models[name]
        if hasattr(model_obj, "feature_importances_"):
            st.write(f"**{name}**")
            importance_df = pd.DataFrame(
                {"Feature": FEATURE_ORDER, "Importance": model_obj.feature_importances_}
            ).sort_values("Importance", ascending=False).set_index("Feature")
            st.bar_chart(importance_df)
        else:
            st.caption(f"{name}: feature importance not available.")

# ---------------------------------------------------------------------
# Build the feature row — exact order, exact saved encoders, no refitting
# ---------------------------------------------------------------------
def build_input_row():
    raw = dict(aq_answers)
    raw.update(
        {
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "jaundice": jaundice,
            "austim": austim,
            "contry_of_res": contry_of_res,
            "used_app_before": used_app_before,
            # "result": result,
            "result": suggested_result,
            "relation": relation,
        }
    )

    # Transform categorical values with the SAME encoders fit during training.
    for col in CATEGORICAL_COLUMNS:
        raw[col] = encoders[col].transform([raw[col]])[0]

    row = pd.DataFrame([[raw[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return row


def predict_with(model_name, row):
    model = models[model_name]
    pred = model.predict(row)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0][1]  # P(class == 1, i.e. ASD)
    return pred, proba


def label_for(pred):
    return "ASD Detected" if pred == 1 else "No ASD Detected"


st.divider()

if st.button("Predict", type="primary"):
    row = build_input_row()

    if model_choice == "Compare All Models":
        st.subheader("Results — All Models")
        results = []
        for name in ["Decision Tree", "Random Forest", "XGBoost"]:
            pred, proba = predict_with(name, row)
            results.append(
                {
                    "Model": name,
                    "Prediction": label_for(pred),
                    "ASD Probability": f"{proba:.1%}" if proba is not None else "N/A",
                }
            )
        st.table(pd.DataFrame(results).set_index("Model"))
    else:
        pred, proba = predict_with(model_choice, row)
        result_label = label_for(pred)
        if pred == 1:
            st.error(f"**{result_label}**")
        else:
            st.success(f"**{result_label}**")
        if proba is not None:
            st.metric("Predicted probability of ASD", f"{proba:.1%}")

    st.caption(
        "Reminder: this tool is for educational demonstration only and is "
        "not a medical diagnosis."
    )
