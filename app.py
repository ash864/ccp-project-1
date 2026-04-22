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


def download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def metric_card(label: str, value: str, tone: str = "teal") -> None:
    st.markdown(
        f"""
        <div class="metric-card metric-{tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_panel(title: str, count: int, description: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="decision-panel panel-{tone}">
            <div class="decision-count">{count}</div>
            <div>
                <div class="decision-title">{title}</div>
                <div class="decision-copy">{description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;600;700;800&display=swap');
    :root {
        --ink: #10201f;
        --muted: #5d6b68;
        --paper: rgba(255, 255, 255, 0.82);
        --teal: #0f766e;
        --green: #15803d;
        --amber: #b45309;
        --red: #b91c1c;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(15, 118, 110, 0.24), transparent 28%),
            radial-gradient(circle at 86% 10%, rgba(245, 158, 11, 0.22), transparent 26%),
            radial-gradient(circle at 68% 86%, rgba(30, 64, 175, 0.10), transparent 28%),
            linear-gradient(135deg, #f8f3e8 0%, #eef8f4 46%, #fffaf0 100%);
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 34px 38px;
        border-radius: 30px;
        background:
            linear-gradient(135deg, rgba(8, 47, 73, 0.96), rgba(15, 118, 110, 0.92)),
            linear-gradient(90deg, #0f766e, #1d4ed8);
        color: white;
        box-shadow: 0 26px 70px rgba(15, 23, 42, 0.22);
        margin-bottom: 24px;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -80px;
        top: -90px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: clamp(2.3rem, 5vw, 4.4rem);
        letter-spacing: -0.08em;
        line-height: 0.95;
    }
    .hero p {
        max-width: 760px;
        margin: 0;
        font-size: 1.06rem;
        opacity: 0.96;
    }
    .eyebrow {
        display: inline-block;
        margin-bottom: 14px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.18);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .glass-card {
        background: var(--paper);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.10);
        backdrop-filter: blur(12px);
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin: 18px 0 10px 0;
        letter-spacing: -0.03em;
    }
    .hint {
        color: var(--muted);
        font-size: 0.94rem;
        margin-bottom: 10px;
    }
    .metric-card {
        min-height: 112px;
        background: rgba(255, 255, 255, 0.90);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 700;
    }
    .metric-value {
        color: var(--ink);
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-top: 6px;
    }
    .metric-red .metric-value { color: var(--red); }
    .metric-amber .metric-value { color: var(--amber); }
    .metric-green .metric-value { color: var(--green); }
    .metric-teal .metric-value { color: var(--teal); }
    .decision-panel {
        display: flex;
        gap: 16px;
        align-items: center;
        min-height: 120px;
        border-radius: 26px;
        padding: 22px;
        color: white;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.14);
    }
    .panel-red { background: linear-gradient(135deg, #991b1b, #ef4444); }
    .panel-amber { background: linear-gradient(135deg, #92400e, #f59e0b); }
    .panel-green { background: linear-gradient(135deg, #166534, #22c55e); }
    .decision-count {
        min-width: 72px;
        height: 72px;
        display: grid;
        place-items: center;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.18);
        font-size: 2rem;
        font-weight: 800;
    }
    .decision-title {
        font-size: 1.2rem;
        font-weight: 800;
    }
    .decision-copy {
        opacity: 0.92;
        font-size: 0.94rem;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 22px;
        border: 1px dashed rgba(15, 118, 110, 0.45);
        background: rgba(255, 255, 255, 0.58);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 10px 18px;
        background: rgba(255, 255, 255, 0.66);
    }
    .stDataFrame {
        border-radius: 18px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Credit Risk Dashboard</div>
        <h1>Loan Default Prediction</h1>
        <p>Upload an Indian bank loan dataset, train multiple models, and turn risk probabilities into clear approve, review, and reject decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([1.05, 0.95], gap="large")

with top_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">Use a CSV with customer loan details and a binary Default column.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload your loan CSV file",
        type=["csv"],
        label_visibility="collapsed",
        help="Upload the dataset with a Default column.",
    )
    train_button = st.button(
        "Train Model And Show Decisions",
        type="primary",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Decision Ranges</div>', unsafe_allow_html=True)
    st.write("Approve: default probability below `0.40`")
    st.write("Manual Review: probability from `0.40` to `0.70`")
    st.write("Reject: probability above `0.70`")
    st.caption("These thresholds can be adjusted later if you want a stricter or safer bank policy.")
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None and train_button:
    dataframe = pd.read_csv(uploaded_file)

    with st.spinner("Training models and creating decision lists..."):
        result = train_from_dataframe(dataframe)

    predictions = result["prediction_frame"].sort_values(
        "Default_Probability",
        ascending=False,
    )
    rejected = predictions[predictions["Loan_Decision"] == "Reject"]
    manual_review = predictions[predictions["Loan_Decision"] == "Manual Review"]
    approve = predictions[predictions["Loan_Decision"] == "Approve"]

    metrics = result["metrics"]

    st.markdown('<div class="section-title">Model Snapshot</div>', unsafe_allow_html=True)
    metric_columns = st.columns(5)
    metric_items = [
        ("Best Model", result["model_name"], "teal"),
        ("Accuracy", f"{metrics['accuracy']:.4f}", "green"),
        ("ROC-AUC", f"{metrics['roc_auc']:.4f}", "teal"),
        ("Precision", f"{metrics['precision']:.4f}", "amber"),
        ("Recall", f"{metrics['recall']:.4f}", "red"),
    ]

    for column, (label, value, tone) in zip(metric_columns, metric_items):
        with column:
            metric_card(label, value, tone)

    st.markdown('<div class="section-title">Decision Queue</div>', unsafe_allow_html=True)
    decision_columns = st.columns(3)
    with decision_columns[0]:
        decision_panel(
            "Reject",
            len(rejected),
            "High-risk applications. These should not be approved without strong additional evidence.",
            "red",
        )
    with decision_columns[1]:
        decision_panel(
            "Manual Review",
            len(manual_review),
            "Borderline applications that need human verification before approval.",
            "amber",
        )
    with decision_columns[2]:
        decision_panel(
            "Approve",
            len(approve),
            "Lower-risk applications according to the model and current threshold.",
            "green",
        )

    st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
    distribution = (
        predictions["Loan_Decision"]
        .value_counts()
        .reindex(["Reject", "Manual Review", "Approve"], fill_value=0)
        .reset_index()
    )
    distribution.columns = ["Decision", "Customers"]
    st.bar_chart(distribution, x="Decision", y="Customers", color="#0f766e")

    tab_reject, tab_review, tab_all = st.tabs(
        ["Rejected Customers", "Manual Review Customers", "All Predictions"]
    )

    with tab_reject:
        st.markdown('<div class="section-title">Customers To Reject</div>', unsafe_allow_html=True)
        st.caption("Sorted by highest default probability first.")
        st.dataframe(rejected, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Rejected Customers CSV",
            data=download_csv(rejected),
            file_name="rejected_customers.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_review:
        st.markdown('<div class="section-title">Customers Needing Manual Review</div>', unsafe_allow_html=True)
        st.caption("These applications are not automatic approvals or automatic rejects.")
        st.dataframe(manual_review, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Manual Review Customers CSV",
            data=download_csv(manual_review),
            file_name="manual_review_customers.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_all:
        st.markdown('<div class="section-title">All Predictions</div>', unsafe_allow_html=True)
        st.dataframe(predictions, use_container_width=True, hide_index=True)
        st.download_button(
            "Download All Predictions CSV",
            data=download_csv(predictions),
            file_name="all_loan_decisions.csv",
            mime="text/csv",
            use_container_width=True,
        )

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

    st.download_button(
        "Download Trained Model",
        data=model_buffer.getvalue(),
        file_name="best_loan_default_model.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )

elif train_button:
    st.warning("Upload the CSV first, then click the training button.")
