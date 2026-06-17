"""
src/train_model.py
---------------------
Trains two classification models (Logistic Regression as a baseline, and
a Random Forest as the main model), compares them, and saves the
better-performing model along with the preprocessing objects needed to
reuse it later (the scaler and the exact list of training feature columns).

Run directly with:
    python src/train_model.py
"""

import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import NUMERIC_COLUMNS, split_features_target  # noqa: E402

DATA_PATH = "data/customer_churn_data.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


def load_and_prepare_data():
    """Loads the raw CSV, cleans it, and encodes it into model-ready
    features and the target column."""
    df = pd.read_csv(DATA_PATH)
    X, y, feature_columns = split_features_target(df)
    return X, y, feature_columns


def scale_numeric_features(X_train, X_test):
    """Fits a StandardScaler on the training data only (to avoid leaking
    information from the test set) and applies it to both splits."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])
    return X_train, X_test, scaler


def evaluate_model(name, model, X_test, y_test):
    """Computes the standard classification metrics and prints a short
    report. Returns a dict of metrics so we can compare models afterwards."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n--- {name} ---")
    for key, value in metrics.items():
        if key != "model":
            print(f"{key:10s}: {value:.3f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics, y_pred


def save_confusion_matrix(y_test, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def save_feature_importance(model, feature_columns, out_dir, top_n=12):
    """Plots and saves the top N most important features from the Random
    Forest. This is one of the easiest, most interview-friendly charts to
    talk through, so it's worth generating even though it's optional."""
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top_features = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=top_features.values, y=top_features.index, color="#1D3557")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Loading and preparing data...")
    X, y, feature_columns = load_and_prepare_data()

    # 80/20 train-test split, stratified so the churn ratio stays consistent
    # in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test, scaler = scale_numeric_features(X_train, X_test)

    # --- Baseline model: Logistic Regression ---
    print("\nTraining Logistic Regression (baseline)...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    log_reg_metrics, _ = evaluate_model("Logistic Regression", log_reg, X_test, y_test)

    # --- Main model: Random Forest, lightly tuned with a small grid search ---
    print("\nTuning Random Forest with GridSearchCV (3-fold CV)...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 10, None],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    rf_model = grid_search.best_estimator_
    print(f"Best Random Forest params: {grid_search.best_params_}")

    rf_metrics, rf_pred = evaluate_model("Random Forest", rf_model, X_test, y_test)

    # --- Pick the better model by F1 score (a balanced metric for an
    # imbalanced classification problem like churn) ---
    if rf_metrics["f1_score"] >= log_reg_metrics["f1_score"]:
        best_model, best_name, best_metrics, best_pred = rf_model, "Random Forest", rf_metrics, rf_pred
    else:
        best_model, best_name, best_metrics, best_pred = log_reg, "Logistic Regression", log_reg_metrics, _

    print(f"\nBest model selected: {best_name}")

    # --- Save everything needed to reproduce predictions later ---
    joblib.dump(best_model, os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    joblib.dump(best_name, os.path.join(MODELS_DIR, "model_name.pkl"))

    # --- Save evaluation charts & a metrics summary file ---
    save_confusion_matrix(y_test, best_pred, best_name, OUTPUTS_DIR)
    if best_name == "Random Forest":
        save_feature_importance(rf_model, feature_columns, OUTPUTS_DIR)

    with open(os.path.join(OUTPUTS_DIR, "model_metrics.txt"), "w") as f:
        f.write("Customer Churn Prediction - Model Evaluation Summary\n")
        f.write("=" * 55 + "\n\n")
        for metrics in [log_reg_metrics, rf_metrics]:
            f.write(f"{metrics['model']}:\n")
            for key, value in metrics.items():
                if key != "model":
                    f.write(f"  {key}: {value:.3f}\n")
            f.write("\n")
        f.write(f"Selected model for deployment: {best_name}\n")

    print(f"\nSaved model + preprocessing objects to '{MODELS_DIR}/'")
    print(f"Saved evaluation charts + metrics summary to '{OUTPUTS_DIR}/'")


if __name__ == "__main__":
    main()