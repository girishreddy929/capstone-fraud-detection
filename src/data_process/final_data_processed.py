# src/final_dataset/final_explained_dataset.py

import pandas as pd
from src.data_loader.load_fraud_output import load_and_validate_fraud_output
from src.explanation.llm_narrative_openai import generate_explanations_for_df_openai
from src.explanation.shap_integration import compute_shap_values, get_top_features_df

def create_final_explained_dataset(
    csv_path: str,
    openai_model=None,
    shap_model=None,
    top_n_shap: int = 5
) -> pd.DataFrame:
    """
    Generate final explained dataset combining Tasks 1-4.

    Args:
        csv_path (str): Path to raw fraud model CSV
        openai_model: OpenAI model for Task 3 explanations
        shap_model: ML model for SHAP values (Task 4)
        top_n_shap (int): Number of top SHAP features per transaction

    Returns:
        pd.DataFrame: Final explained dataset
    """

    # Task 1: Load & validate dataset
    df = load_and_validate_fraud_output(csv_path)

    # Task 3: LLM explanations
    df = generate_explanations_for_df_openai(df)

    # Task 4: SHAP top features
    if shap_model:
        feature_cols = [col for col in df.columns if col not in ["transaction_id", "fraud_score", "fraud_prediction", "explanation"]]
        X = df[feature_cols]
        shap_values = compute_shap_values(shap_model, X)
        top_features_df = get_top_features_df(shap_values, X, top_n=top_n_shap)
        df = pd.concat([df.reset_index(drop=True), top_features_df.drop(columns=["transaction_index"])], axis=1)

    # Task 2: Rule-based factors
    from src.explanation.llm_narrative_openai import generate_rule_based_reasons
    def get_rule_based_factors(row: pd.Series) -> str:
        reasons = generate_rule_based_reasons(row)
        # Convert reason keys to human-friendly labels
        label_map = {
            "high_transaction_amount": "High Transaction Amount",
            "geo_mismatch": "Geo Mismatch",
            "device_fingerprint_changed": "Device Fingerprint Changed",
            "high_velocity_flag": "High Velocity",
            "high_fraud_score": "High Fraud Score",
            "large_amount_geo_mismatch": "Large Amount + Geo Mismatch"
        }
        return ", ".join([label_map.get(r, r) for r in reasons]) or "No notable patterns"

    df["rule_based_factors"] = df.apply(get_rule_based_factors, axis=1)

    # Reorder columns for readability
    final_cols = [
        "transaction_id", "fraud_score", "fraud_prediction", "explanation", "rule_based_factors"
    ]
    shap_cols = [c for c in df.columns if "top_feature" in c]
    final_cols.extend(shap_cols)
    # Include remaining original columns
    original_cols = [c for c in df.columns if c not in final_cols]
    final_cols.extend(original_cols)

    df_final = df[final_cols]
    return df_final


if __name__ == "__main__":
    csv_path = "data/raw/fraud_model_output.csv"
    df_final = create_final_explained_dataset(csv_path)
    print(df_final.head())
    df_final.to_csv("data/processed/fraud_model_processed.csv", index=False)
