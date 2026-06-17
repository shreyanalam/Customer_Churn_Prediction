"""
generate_dataset.py
--------------------
Generates a synthetic telecom customer churn dataset and saves it to
data/customer_churn_data.csv.

A generated dataset is used instead of an external download so the
project remains fully self-contained and reproducible: running this
script always produces the same data, with no dependency on an outside
link that could change or break.

The columns and value ranges are modeled on the structure of the public
Telco Customer Churn dataset, so this file could later be replaced with
the real Kaggle version without requiring changes to the rest of the
project.
"""

import numpy as np
import pandas as pd

# Fixing the random seed makes the generated dataset identical every time
# this script is run -- important for reproducibility.
np.random.seed(42)

N = 1500  # number of customer records to generate


def generate_customer_data(n_rows: int) -> pd.DataFrame:
    """Builds a DataFrame of synthetic customer records with realistic
    relationships between features and the churn outcome."""

    customer_id = [f"CUST-{1000 + i}" for i in range(n_rows)]
    gender = np.random.choice(["Male", "Female"], n_rows)
    senior_citizen = np.random.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n_rows, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n_rows, p=[0.30, 0.70])

    # Tenure (months with the company) - skewed towards shorter and longer
    # tenures, similar to real telecom data.
    tenure = np.random.choice(
        np.concatenate([np.arange(0, 12), np.arange(12, 72)]),
        n_rows,
        p=None,
    )
    tenure = np.clip(np.random.exponential(scale=24, size=n_rows).astype(int), 0, 72)

    phone_service = np.random.choice(["Yes", "No"], n_rows, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        np.random.choice(["Yes", "No"], n_rows, p=[0.45, 0.55]),
    )

    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n_rows, p=[0.35, 0.45, 0.20]
    )

    def internet_dependent_feature(p_yes=0.4):
        """Helper: a feature that only makes sense if the customer has
        internet service; otherwise it is 'No internet service'."""
        result = np.where(
            internet_service == "No",
            "No internet service",
            np.random.choice(["Yes", "No"], n_rows, p=[p_yes, 1 - p_yes]),
        )
        return result

    online_security = internet_dependent_feature(0.35)
    online_backup = internet_dependent_feature(0.40)
    device_protection = internet_dependent_feature(0.40)
    tech_support = internet_dependent_feature(0.35)
    streaming_tv = internet_dependent_feature(0.45)
    streaming_movies = internet_dependent_feature(0.45)

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.24, 0.21]
    )
    paperless_billing = np.random.choice(["Yes", "No"], n_rows, p=[0.59, 0.41])
    payment_method = np.random.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges depend on which services the customer has subscribed to.
    base_charge = np.full(n_rows, 18.0)
    base_charge += np.where(phone_service == "Yes", np.random.uniform(5, 20, n_rows), 0)
    base_charge += np.where(internet_service == "DSL", np.random.uniform(15, 35, n_rows), 0)
    base_charge += np.where(internet_service == "Fiber optic", np.random.uniform(35, 65, n_rows), 0)
    for feature in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
        base_charge += np.where(feature == "Yes", np.random.uniform(3, 8, n_rows), 0)

    monthly_charges = np.round(base_charge, 2)
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(-20, 20, n_rows), 2)
    total_charges = np.clip(total_charges, 0, None)

   # --- Churn probability model ---
    # Churn risk increases with month-to-month contracts, high monthly
    # charges, low tenure, fiber-optic service without security/support
    # add-ons, and electronic check payments -- patterns consistent with
    # the real Telco churn dataset.
    churn_score = np.zeros(n_rows)
    churn_score += np.where(contract == "Month-to-month", 0.35, 0)
    churn_score += np.where(contract == "One year", 0.10, 0)
    churn_score += np.where(internet_service == "Fiber optic", 0.20, 0)
    churn_score += np.where(tech_support == "No", 0.10, 0)
    churn_score += np.where(online_security == "No", 0.10, 0)
    churn_score += np.where(payment_method == "Electronic check", 0.15, 0)
    churn_score += np.where(senior_citizen == 1, 0.05, 0)
    churn_score += (monthly_charges - monthly_charges.mean()) / monthly_charges.std() * 0.08
    churn_score -= (tenure - tenure.mean()) / tenure.std() * 0.15
    churn_score += np.random.normal(0, 0.12, n_rows)  # noise so it's not a perfect rule

    churn_probability = 1 / (1 + np.exp(-4 * (churn_score - 0.74)))  # squash to 0-1, calibrated for a realistic ~27% churn rate
    churn = np.where(np.random.uniform(0, 1, n_rows) < churn_probability, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

   # Inject a small number of missing values into TotalCharges, mirroring
    # a known quirk in the real dataset, so the cleaning step has an
    # actual missing-value case to handle.
    missing_idx = np.random.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan

    return df


if __name__ == "__main__":
    df = generate_customer_data(N)
    output_path = "data/customer_churn_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows -> saved to {output_path}")
    print(f"Churn rate in generated data: {(df['Churn'] == 'Yes').mean():.2%}")