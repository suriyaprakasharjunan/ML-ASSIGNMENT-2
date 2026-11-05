from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "color",
]
TARGET_COLUMN = "quality"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def calculate_metrics(model, X: pd.DataFrame, y_true: pd.Series):
    y_pred = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (weighted)": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall (weighted)": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1 Score (weighted)": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    # All five saved classifiers were trained with probability support or
    # expose predict_proba, so calculate weighted one-vs-rest multiclass AUC.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        class_labels = model.classes_
        metrics["AUC (weighted OVR)"] = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="weighted",
            labels=class_labels,
        )
    else:
        metrics["AUC (weighted OVR)"] = float("nan")

    return y_pred, metrics


st.set_page_config(
    page_title="Wine Quality Classification",
    page_icon="🍷",
    layout="wide",
)

st.title("Wine Quality Classification")
st.write(
    "Evaluate one of the five pre-trained classification models on a labelled "
    "CSV test dataset. The application loads saved pipelines; it does not "
    "retrain models."
)

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    st.caption("Upload a CSV containing all 12 input features and the quality target.")

st.subheader("Test data")
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded_file is None:
    if TEST_DATA_PATH.exists():
        st.info("No file uploaded. The bundled test_data.csv is being used.")
        data = pd.read_csv(TEST_DATA_PATH)
    else:
        st.warning("Upload a labelled CSV file to begin evaluation.")
        st.stop()
else:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
missing_columns = [column for column in required_columns if column not in data.columns]

if missing_columns:
    st.error("The CSV is missing required columns: " + ", ".join(missing_columns))
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

evaluation_data = data[required_columns].copy()
if evaluation_data.isnull().any().any():
    null_columns = evaluation_data.columns[evaluation_data.isnull().any()].tolist()
    st.error("Missing values were found in: " + ", ".join(null_columns))
    st.stop()

X_test = evaluation_data[FEATURE_COLUMNS]
y_test = evaluation_data[TARGET_COLUMN]

model_path = MODEL_DIR / MODEL_FILES[selected_model]
if not model_path.exists():
    st.error(f"Saved model file not found: {model_path.name}")
    st.stop()

try:
    model = load_model(str(model_path))
    predictions, metrics = calculate_metrics(model, X_test, y_test)
except Exception as exc:
    st.error(
        "Model evaluation failed. Confirm that the installed scikit-learn "
        "version matches the version used to train the saved models "
        "(1.9.1), and check that the CSV contains valid feature values."
    )
    st.exception(exc)
    st.stop()

st.subheader(f"Evaluation results — {selected_model}")
st.caption(f"Evaluated {len(evaluation_data):,} labelled rows.")

metric_columns = st.columns(3)
metric_order = [
    "Accuracy",
    "AUC (weighted OVR)",
    "Precision (weighted)",
    "Recall (weighted)",
    "F1 Score (weighted)",
    "MCC",
]
for index, metric_name in enumerate(metric_order):
    metric_columns[index % 3].metric(metric_name, f"{metrics[metric_name]:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    labels = sorted(pd.unique(pd.concat([y_test, pd.Series(predictions)])))
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

with st.expander("Preview evaluated data"):
    preview = evaluation_data.copy()
    preview["Predicted quality"] = predictions
    st.dataframe(preview.head(20), use_container_width=True)

st.caption(
    "Note: metrics are calculated against the quality labels included in the "
    "uploaded test CSV. Do not upload training data when reporting test performance."
)
