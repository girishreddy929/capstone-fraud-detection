# src/data_process/vizualization_reporting.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Create directories for saving reports
# -----------------------------
REPORT_DIR = os.path.join(os.getcwd(), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# Plot fraud score distribution
# -----------------------------
def plot_fraud_score_distribution(df: pd.DataFrame):
    if "fraud_score" not in df.columns:
        print("Warning: 'fraud_score' column not found. Skipping plot.")
        return

    plt.figure(figsize=(8,5))
    sns.histplot(df["fraud_score"], bins=20, kde=True, color="skyblue")
    plt.title("Fraud Score Distribution")
    plt.xlabel("Fraud Score")
    plt.ylabel("Count")
    output_path = os.path.join(REPORT_DIR, "fraud_score_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved fraud score distribution plot: {output_path}")

# -----------------------------
# Plot fraud prediction counts
# -----------------------------
def plot_fraud_prediction_counts(df: pd.DataFrame):
    if "fraud_prediction" not in df.columns:
        print("Warning: 'fraud_prediction' column not found. Skipping plot.")
        return

    plt.figure(figsize=(6,4))
    sns.countplot(x="fraud_prediction", data=df, palette="Set2")
    plt.title("Fraud Prediction Counts")
    plt.xlabel("Fraud Prediction (0=Non-Fraud, 1=Fraud)")
    plt.ylabel("Count")
    output_path = os.path.join(REPORT_DIR, "fraud_prediction_counts.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved fraud prediction counts plot: {output_path}")

# -----------------------------
# Plot rule-based factors
# -----------------------------
def plot_rule_based_factors(df: pd.DataFrame):
    if "rule_based_factors" not in df.columns:
        print("Warning: 'rule_based_factors' column not found. Skipping plot.")
        return

    factor_counts = df["rule_based_factors"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=factor_counts.values, y=factor_counts.index, palette="viridis")
    plt.title("Rule-Based Factors Frequency")
    plt.xlabel("Count")
    plt.ylabel("Rule-Based Factors")
    output_path = os.path.join(REPORT_DIR, "rule_based_factors.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved rule-based factors plot: {output_path}")

# -----------------------------
# Plot SHAP top features if available
# -----------------------------
def plot_top_shap_features(df: pd.DataFrame):
    shap_cols = [c for c in df.columns if "top_feature" in c]
    if not shap_cols:
        print("Warning: No SHAP top feature columns found. Skipping plot.")
        return

    for col in shap_cols:
        plt.figure(figsize=(8,4))
        sns.countplot(x=col, data=df, palette="magma")
        plt.title(f"Top SHAP Feature: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        output_path = os.path.join(REPORT_DIR, f"{col}_distribution.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved SHAP feature plot: {output_path}")

# -----------------------------
# Main reporting function
# -----------------------------
def generate_reports(df: pd.DataFrame, shap_values=None):
    print("Generating reports...")
    plot_fraud_score_distribution(df)
    plot_fraud_prediction_counts(df)
    plot_rule_based_factors(df)
    plot_top_shap_features(df)
    print("All reports generated.")

# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    # Load processed dataset
    csv_path = "data/processed/fraud_model_processed.csv"
    if not os.path.exists(csv_path):
        print(f"Processed CSV not found at {csv_path}. Please run final_explained_dataset.py first.")
    else:
        df_final = pd.read_csv(csv_path)
        generate_reports(df_final)
