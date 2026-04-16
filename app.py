from __future__ import annotations

from io import BytesIO

import joblib
import pandas as pd
import streamlit as st

from bank_loan_default_prediction_india import train_from_dataframe


st.set_page_config(
    page_title="Bank Loan Default Prediction",
    page_icon="IN",
    layout="wide",
)


def build_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(245, 158, 11, 0.18), transparent 30%),
            linear-gradient(135deg, #f7f4ea 0%, #eef7f5 45%, #fdfcf8 100%);
    }
    .hero {
        padding: 28px 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.95), rgba(21, 128, 61, 0.88));
        color: white;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
        margin-bottom: 20px;
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 2.3rem;
    }
    .hero p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.96;
    }
    .card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 18px 20px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        text-align: center;
    }
    .metric-label {
        color: #0f172a;
        font-size: 0.92rem;
        opacity: 0.72;
    }
    .metric-value {
        color: #0f766e;
        font-size: 1.7rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Bank Loan Default Prediction</h1>
        <p>Upload your Indian bank loan CSV, train multiple models, compare performance, and download predictions in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose the loan CSV file",
        type=["csv"],
        help="Upload the dataset containing a binary default column such as Default.",
    )
    train_button = st.button("Train Model", type="primary", use_container_width=True)
    st.caption("The app compares Logistic Regression, Random Forest, and Gradient Boosting, then keeps the best model by ROC-AUC.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What You Get")
    st.write("1. Automatic target-column detection")
    st.write("2. Data preprocessing for numeric and categorical fields")
    st.write("3. Model comparison with key performance metrics")
    st.write("4. Downloadable predictions and trained model")
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None and train_button:
    dataframe = pd.read_csv(uploaded_file)
    with st.spinner("Training models and generating predictions..."):
        result = train_from_dataframe(dataframe)

    metrics = result["metrics"]
    prediction_frame = result["prediction_frame"]

    st.subheader("Best Model Performance")
    metric_columns = st.columns(5)
    metric_items = [
        ("Best Model", result["model_name"]),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ("Precision", f"{metrics['precision']:.4f}"),
        ("Recall", f"{metrics['recall']:.4f}"),
        ("ROC-AUC", f"{metrics['roc_auc']:.4f}"),
    ]
    for column, (label, value) in zip(metric_columns, metric_items):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    compare_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Accuracy": round(model_metrics["accuracy"], 4),
                "Precision": round(model_metrics["precision"], 4),
                "Recall": round(model_metrics["recall"], 4),
                "F1 Score": round(model_metrics["f1_score"], 4),
                "ROC-AUC": round(model_metrics["roc_auc"], 4),
            }
            for model_name, model_metrics in result["all_results"]
        ]
    )

    detail_left, detail_right = st.columns([1, 1], gap="large")

    with detail_left:
        st.subheader("Model Comparison")
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        st.subheader("Classification Report")
        st.code(result["classification_report"])

    with detail_right:
        st.subheader("Predictions Preview")
        st.dataframe(prediction_frame.head(20), use_container_width=True)

        st.subheader("Confusion Matrix")
        matrix = result["confusion_matrix"]
        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.dataframe(matrix_df, use_container_width=True)

    model_buffer = BytesIO()
    joblib.dump(
        {
            "model_name": result["model_name"],
            "model": result["model"],
            "target_column": result["target_column"],
            "feature_columns": result["feature_columns"],
        },
        model_buffer,
    )

    download_left, download_right, download_summary = st.columns(3)
    with download_left:
        st.download_button(
            "Download Predictions CSV",
            data=build_download_bytes(prediction_frame),
            file_name="loan_default_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_right:
        st.download_button(
            "Download Trained Model",
            data=model_buffer.getvalue(),
            file_name="best_loan_default_model.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with download_summary:
        st.download_button(
            "Download Summary Report",
            data=result["summary_text"].encode("utf-8"),
            file_name="loan_default_model_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
elif train_button:
    st.warning("Upload a CSV file first, then click Train Model.")
