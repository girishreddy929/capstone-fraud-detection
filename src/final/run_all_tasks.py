# run_all_tasks.py
# """
# Complete pipeline for Fraud Detection Explanation Assistant
# Tasks 1-7:
# 1️⃣ Load & validate raw fraud data
# 2️⃣ Generate rule-based reasons
# 3️⃣ Generate LLM explanations (OpenAI)
# 4️⃣ Compute SHAP values (optional)
# 5️⃣ Create final explained dataset
# 6️⃣ Visualization & Reporting (optional)
# 7️⃣ SME feedback & human evaluation loop
# """

import os
import pandas as pd
from src.data_loader.load_fraud_output import load_and_validate_fraud_output
from src.explanation.llm_narrative_openai import (
    generate_explanations_for_df_openai,
    generate_rule_based_reasons,
)
from src.explanation.shap_integration import compute_shap_values, get_top_features_df
# Optional: Task 6 visualization imports
from src.data_process.vizualization_reporting import generate_reports
from src.data_process.feedback_system import collect_sme_feedback, summarize_feedback, integrate_feedback

# -----------------------------
# Config
# -----------------------------
RAW_CSV = "data/raw/fraud_model_output.csv"
PROCESSED_CSV = "data/processed/fraud_model_processed.csv"
FINAL_FEEDBACK_CSV = "data/final/fraud_explainability.csv"
OPENAI_MODEL = "gpt-4o-mini"       # Defaults to config DEFAULT_MODEL
SHAP_MODEL = None         # Provide your ML model if SHAP is needed
TOP_N_SHAP = 5

# -----------------------------
# 1️⃣ Load & validate dataset
# -----------------------------
print("🔹 Task 1: Loading and validating raw data...")
df_validated = load_and_validate_fraud_output(RAW_CSV)

# -----------------------------
# 3️⃣ Generate OpenAI explanations
# -----------------------------
print("🔹 Task 3: Generating LLM explanations...")
df_explained = generate_explanations_for_df_openai(df_validated, model=OPENAI_MODEL)

# -----------------------------
# 4️⃣ SHAP feature importance (optional)
# -----------------------------
if SHAP_MODEL:
    print("🔹 Task 4: Computing SHAP values and top features...")
    feature_cols = [c for c in df_explained.columns if c not in ["transaction_id", "fraud_score", "fraud_prediction", "explanation"]]
    X = df_explained[feature_cols]
    shap_values = compute_shap_values(SHAP_MODEL, X)
    top_features_df = get_top_features_df(shap_values, X, top_n=TOP_N_SHAP)
    df_explained = pd.concat([df_explained.reset_index(drop=True), top_features_df.drop(columns=["transaction_index"])], axis=1)
else:
    shap_values = None

# -----------------------------
# 2️⃣ Rule-based factors
# -----------------------------
print("🔹 Task 2: Computing rule-based factors...")
def get_rule_based_factors(row: pd.Series) -> str:
    reasons = generate_rule_based_reasons(row)
    label_map = {
        "high_transaction_amount": "High Transaction Amount",
        "geo_mismatch": "Geo Mismatch",
        "device_fingerprint_changed": "Device Fingerprint Changed",
        "high_velocity_flag": "High Velocity",
        "high_fraud_score": "High Fraud Score",
        "large_amount_geo_mismatch": "Large Amount + Geo Mismatch"
    }
    return ", ".join([label_map.get(r, r) for r in reasons]) or "No notable patterns"

df_explained["rule_based_factors"] = df_explained.apply(get_rule_based_factors, axis=1)

# -----------------------------
# 5️⃣ Create final explained dataset
# -----------------------------
print("🔹 Task 5: Creating final explained dataset...")
final_cols = ["transaction_id", "fraud_score", "fraud_prediction", "explanation", "rule_based_factors"]
shap_cols = [c for c in df_explained.columns if "top_feature" in c]
final_cols.extend(shap_cols)
original_cols = [c for c in df_explained.columns if c not in final_cols]
final_cols.extend(original_cols)
df_final = df_explained[final_cols]

# Save intermediate processed dataset
os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
df_final.to_csv(PROCESSED_CSV, index=False)
print(f"Processed dataset saved at {PROCESSED_CSV}")

# -----------------------------
# 6️⃣ Visualization & Reporting (optional)
# -----------------------------
print("🔹 Task 6: Generating reports and visualizations...")
generate_reports(df_final, shap_values=shap_values if SHAP_MODEL else None)

# -----------------------------
# 7️⃣ SME feedback & evaluation loop
# -----------------------------
print("🔹 Task 7: Collecting SME feedback and integrating...")
feedback_df = collect_sme_feedback(df_final)
summary_metrics = summarize_feedback(feedback_df)
df_final_with_feedback = integrate_feedback(df_final, feedback_df)

# Save final dataset with feedback
os.makedirs(os.path.dirname(FINAL_FEEDBACK_CSV), exist_ok=True)
df_final_with_feedback.to_csv(FINAL_FEEDBACK_CSV, index=False)
print(f"Final dataset with SME feedback saved at {FINAL_FEEDBACK_CSV}")
