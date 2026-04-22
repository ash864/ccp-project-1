from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(r"C:\Users\lanzd\Downloads\files\indian_bank_loan_dataset.csv")
MODEL_OUTPUT = Path("best_loan_default_model.joblib")
PREDICTION_OUTPUT = Path("loan_default_predictions.csv")
SUMMARY_OUTPUT = Path("loan_default_model_summary.txt")

COMMON_TARGET_NAMES = (
    "default",
    "loan_default",
    "defaulted",
    "target",
    "class",
    "label",
    "loan_status",
)


def infer_target_column(columns: Iterable[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    for name in COMMON_TARGET_NAMES:
        if name in lowered:
            return lowered[name]
    raise ValueError(
        "Could not infer the target column from the dataset columns."
    )


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        unique_values = sorted(series.dropna().unique().tolist())
        if set(unique_values).issubset({0, 1}):
            return series.astype(int)
        if len(unique_values) == 2:
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            return series.map(mapping).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    positive_tokens = {"1", "yes", "y", "true", "default", "bad", "defaulter"}
    negative_tokens = {"0", "no", "n", "false", "paid", "good", "non-default"}

    if set(normalized.unique()).issubset(positive_tokens | negative_tokens):
        return normalized.map(
            lambda value: 1 if value in positive_tokens else 0
        ).astype(int)

    unique_values = sorted(normalized.unique().tolist())
    if len(unique_values) != 2:
        raise ValueError("Target column must be binary.")

    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return normalized.map(mapping).astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_model_candidates(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=14,
                        min_samples_split=8,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def add_loan_decisions(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    result = prediction_frame.copy()
    probability = result["Default_Probability"]
    result["Risk_Level"] = pd.cut(
        probability,
        bins=[-1, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"],
    )
    result["Loan_Decision"] = pd.cut(
        probability,
        bins=[-1, 0.4, 0.7, 1],
        labels=["Approve", "Manual Review", "Reject"],
    )

    priority_columns = [
        "Loan_Decision",
        "Risk_Level",
        "Default_Probability",
        "Predicted_Default",
    ]
    ordered_columns = priority_columns + [
        column for column in result.columns if column not in priority_columns
    ]
    return result[ordered_columns]


def train_from_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        raise ValueError("The provided CSV file is empty.")

    target_column = infer_target_column(df.columns)
    X = df.drop(columns=[target_column]).copy()
    y = normalize_target(df[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    model_candidates = build_model_candidates(preprocessor)

    best_name = ""
    best_model = None
    best_metrics = None
    all_results: list[tuple[str, dict[str, float]]] = []

    for model_name, model in model_candidates.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        all_results.append((model_name, metrics))

        if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
            best_name = model_name
            best_model = model
            best_metrics = metrics

    assert best_model is not None
    assert best_metrics is not None

    final_predictions = best_model.predict(X_test)
    final_probabilities = best_model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, final_predictions)
    matrix = confusion_matrix(y_test, final_predictions)

    prediction_frame = X_test.copy()
    prediction_frame["Actual_Default"] = y_test.values
    prediction_frame["Predicted_Default"] = final_predictions
    prediction_frame["Default_Probability"] = final_probabilities
    prediction_frame = add_loan_decisions(prediction_frame)
    prediction_frame.to_csv(PREDICTION_OUTPUT, index=False)

    summary_lines = [
        "Bank Loan Default Prediction (India)",
        "=" * 40,
        f"Dataset: {DATA_PATH}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Target column: {target_column}",
        "",
        "Model comparison (sorted by training loop order):",
    ]

    for model_name, metrics in all_results:
        summary_lines.append(
            f"- {model_name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    summary_lines.extend(
        [
            "",
            f"Best model: {best_name}",
            f"Best Accuracy: {best_metrics['accuracy']:.4f}",
            f"Best Precision: {best_metrics['precision']:.4f}",
            f"Best Recall: {best_metrics['recall']:.4f}",
            f"Best F1 Score: {best_metrics['f1_score']:.4f}",
            f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}",
            "",
            "Decision rules:",
            "Default_Probability below 0.40: Approve",
            "Default_Probability from 0.40 to 0.70: Manual Review",
            "Default_Probability above 0.70: Reject",
            "",
            "Confusion Matrix:",
            str(matrix),
            "",
            "Classification Report:",
            report,
            "",
            f"Saved model file: {MODEL_OUTPUT.resolve()}",
            f"Saved predictions file: {PREDICTION_OUTPUT.resolve()}",
        ]
    )

    return {
        "model_name": best_name,
        "model": best_model,
        "target_column": target_column,
        "feature_columns": X.columns.tolist(),
        "metrics": best_metrics,
        "all_results": all_results,
        "classification_report": report,
        "confusion_matrix": matrix,
        "prediction_frame": prediction_frame,
        "summary_text": "\n".join(summary_lines),
    }


def train_from_csv(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    result = train_from_dataframe(df)
    result["dataset_path"] = str(csv_path)
    return result


def save_training_outputs(result: dict[str, Any]) -> None:
    result["prediction_frame"].to_csv(PREDICTION_OUTPUT, index=False)
    joblib.dump(
        {
            "model_name": result["model_name"],
            "model": result["model"],
            "target_column": result["target_column"],
            "feature_columns": result["feature_columns"],
        },
        MODEL_OUTPUT,
    )
    SUMMARY_OUTPUT.write_text(result["summary_text"], encoding="utf-8")


def main() -> None:
    result = train_from_csv(DATA_PATH)
    save_training_outputs(result)

    print(result["summary_text"])
    print(f"\nSummary saved to: {SUMMARY_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
