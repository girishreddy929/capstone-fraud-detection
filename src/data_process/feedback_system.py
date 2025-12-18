# src/final_dataset/feedback_system.py

import pandas as pd
import numpy as np
# from src.data_process.final_explained_dataset import create_final_explained_dataset

# -----------------------------
# 1️⃣ Collect SME Feedback
# -----------------------------
def collect_sme_feedback(df: pd.DataFrame, feedback_csv: str = "data/final/fraud_explainability_feedback.csv") -> pd.DataFrame:
    """
    Collect or simulate SME ratings for each transaction explanation.

    Args:
        df: DataFrame with final explanations (Task 5 output)
        feedback_csv: Path to save the feedback CSV

    Returns:
        pd.DataFrame: DataFrame with SME feedback
    """
    feedback = df.copy()
    
    # Simulate SME ratings (replace with real input in production)
    feedback["clarity_rating"] = np.random.randint(3, 6, size=len(feedback))        # 3-5 scale
    feedback["accuracy_rating"] = np.random.randint(3, 6, size=len(feedback))
    feedback["actionability_rating"] = np.random.randint(3, 6, size=len(feedback))
    feedback["comments"] = ""  # Optional: SMEs can fill this manually

    # Save feedback CSV for record
    feedback.to_csv(feedback_csv, index=False)
    print(f"SME feedback saved at {feedback_csv}")

    return feedback

# -----------------------------
# 2️⃣ Summarize Feedback
# -----------------------------
def summarize_feedback(feedback_df: pd.DataFrame):
    """
    Summarize SME feedback to compute averages and percentage of high ratings.
    """
    metrics = {}
    for col in ["clarity_rating", "accuracy_rating", "actionability_rating"]:
        metrics[f"{col}_avg"] = feedback_df[col].mean()
        metrics[f"{col}_>=4_pct"] = (feedback_df[col] >= 4).mean() * 100

    print("SME Feedback Summary:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    
    return metrics

# -----------------------------
# 3️⃣ Integrate Feedback with Final Dataset
# -----------------------------
def integrate_feedback(df_final: pd.DataFrame, feedback_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SME feedback into the final explained dataset.
    """
    return df_final.merge(
        feedback_df[["transaction_id", "clarity_rating", "accuracy_rating", "actionability_rating", "comments"]],
        on="transaction_id",
        how="left"
    )

# # -----------------------------
# # 4️⃣ Main Execution
# # -----------------------------
# if __name__ == "__main__":
#     # Step 1: Generate final explained dataset (Tasks 1-5)
#     csv_path = "data/raw/fraud_model_output.csv"
#     df_final = create_final_explained_dataset(csv_path)

#     # Step 2: Collect SME feedback
#     feedback_df = collect_sme_feedback(df_final)

#     # Step 3: Summarize feedback
#     metrics = summarize_feedback(feedback_df)

#     # Step 4: Integrate feedback into final dataset
#     df_with_feedback = integrate_feedback(df_final, feedback_df)

#     # Step 5: Save final dataset with SME feedback
#     output_path = "data/final/fraud_explainability.csv"
#     df_with_feedback.to_csv(output_path, index=False)
#     print(f"Final dataset with SME feedback saved at {output_path}")
