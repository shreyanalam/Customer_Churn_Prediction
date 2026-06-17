"""
src/predict_utils.py
-----------------------
Loads the trained model and preprocessing objects, and exposes a single
predict() method on the ChurnPredictor class that both a command-line
script and the Streamlit app can call. Centralizing this logic in one
place ensures predictions are always generated using the exact same
preprocessing steps that were applied during training, rather than
duplicating that logic across multiple files where it could fall out of
sync.
"""

import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import NUMERIC_COLUMNS, clean_data, encode_features  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class ChurnPredictor:
    """Wraps the saved model + scaler + feature column list so a caller
    only needs to provide raw, human-readable customer details."""

    def __init__(self, models_dir: str = MODELS_DIR):
        model_path = os.path.join(models_dir, "churn_model.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        columns_path = os.path.join(models_dir, "feature_columns.pkl")
        name_path = os.path.join(models_dir, "model_name.pkl")

        for path in [model_path, scaler_path, columns_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing '{path}'. Run 'python src/train_model.py' first "
                    "to train the model before making predictions."
                )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(columns_path)
        self.model_name = joblib.load(name_path) if os.path.exists(name_path) else "Model"

    def predict(self, customer: dict) -> dict:
        """Takes a dict of raw customer attributes (matching the original
        CSV column names, minus 'Churn' and 'customerID') and returns the
        predicted label plus the churn probability.
        """
        df = pd.DataFrame([customer])
        df = clean_data(df)
        X, _ = encode_features(df, feature_columns=self.feature_columns)
        X[NUMERIC_COLUMNS] = self.scaler.transform(X[NUMERIC_COLUMNS])

        probability = float(self.model.predict_proba(X)[0, 1])
        label = "Churn" if probability >= 0.5 else "No Churn"

        return {
            "prediction": label,
            "churn_probability": round(probability, 4),
            "model_used": self.model_name,
        }


if __name__ == "__main__":
    # Manual test entry point: python src/predict_utils.py
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
    }

    predictor = ChurnPredictor()
    result = predictor.predict(sample_customer)
    print("Sample high-risk customer prediction:")
    print(result)