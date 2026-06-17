"""
src/visualize.py
-------------------
Generates exploratory data analysis (EDA) charts and saves them as PNG
files in the outputs/ folder. This separates the visualization and
analysis step from model training, producing a standalone record of the
key patterns in the dataset.

Run directly with:
    python src/visualize.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # render to file, no display window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Allow running this script directly (python src/visualize.py) as well as
# importing it from the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sns.set_style("whitegrid")
PALETTE = {"Yes": "#E63946", "No": "#2A9D8F"}


def ensure_output_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path


def plot_churn_distribution(df, out_dir):
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="Churn", hue="Churn", palette=PALETTE, legend=False)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
        )
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "churn_distribution.png"), dpi=150)
    plt.close()


def plot_tenure_vs_churn(df, out_dir):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, palette=PALETTE, common_norm=False)
    plt.title("Tenure Distribution by Churn Status")
    plt.xlabel("Tenure (months)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tenure_vs_churn.png"), dpi=150)
    plt.close()


def plot_contract_vs_churn(df, out_dir):
    plt.figure(figsize=(6, 4))
    churn_rate = (
        df.groupby("Contract")["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )
    sns.barplot(x=churn_rate.index, y=churn_rate.values, color="#457B9D")
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Contract Type")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "contract_vs_churn.png"), dpi=150)
    plt.close()


def plot_monthly_charges_vs_churn(df, out_dir):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette=PALETTE, legend=False)
    plt.title("Monthly Charges by Churn Status")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "monthly_charges_vs_churn.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df, out_dir):
    numeric_df = df.copy()
    numeric_df["Churn"] = numeric_df["Churn"].map({"Yes": 1, "No": 0})
    numeric_df["SeniorCitizen"] = numeric_df["SeniorCitizen"].astype(int)
    corr = numeric_df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]].corr()

    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def run_all(data_path="data/customer_churn_data.csv", out_dir="outputs"):
    out_dir = ensure_output_dir(out_dir)
    df = pd.read_csv(data_path)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    plot_churn_distribution(df, out_dir)
    plot_tenure_vs_churn(df, out_dir)
    plot_contract_vs_churn(df, out_dir)
    plot_monthly_charges_vs_churn(df, out_dir)
    plot_correlation_heatmap(df, out_dir)

    print(f"Saved 5 EDA charts to '{out_dir}/'")


if __name__ == "__main__":
    run_all()