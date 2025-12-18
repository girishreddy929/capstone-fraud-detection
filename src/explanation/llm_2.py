import os
import pandas as pd
from openai import OpenAI
from src.explanation.templates import ExplanationTemplates
from src.config import OPENAI_API_KEY, DEFAULT_MODEL

# -----------------------------
# Initialize OpenAI client
# -----------------------------
client = OpenAI(
    api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
)

if not client.api_key:
    raise ValueError(
        "OpenAI API key not set. Define OPENAI_API_KEY in src/config.py or environment."
    )

# -----------------------------
# Task 2: Rule-based reasons
# -----------------------------
def generate_rule_based_reasons(row: dict) -> list:
    reasons = []

    try:
        if float(row.get("transaction_amount", 0)) > 5000:
            reasons.append("high_transaction_amount")

        if int(row.get("geo_mismatch", 0)) == 1:
            reasons.append("geo_mismatch")

        if bool(row.get("device_fingerprint_changed", False)):
            reasons.append("device_fingerprint_changed")

        if int(row.get("high_velocity_flag", 0)) == 1:
            reasons.append("high_velocity_flag")

        if float(row.get("fraud_score", 0)) >= 0.8:
            reasons.append("high_fraud_score")

        if float(row.get("transaction_amount", 0)) > 10000 and int(row.get("geo_mismatch", 0)) == 1:
            reasons.append("large_amount_geo_mismatch")

    except Exception as e:
        print(f"Error evaluating rule-based reasons: {e}")

    return reasons

# -----------------------------
# Prompt construction
# -----------------------------
def build_openai_prompt(row: dict, reason_keys: list) -> str:
    templates_text = " ".join([ExplanationTemplates.get_template(k) for k in reason_keys])
    # Convert all values to string to avoid serialization issues
    row_str_dict = {k: str(v) for k, v in row.items()}

    return f"""
You are a fraud analyst assistant.

Generate a concise (1–3 sentences), professional, business-friendly explanation
for why the transaction was flagged as potentially fraudulent.
Avoid speculation and use factual language only.

Transaction features:
{row_str_dict}

Detected risk patterns:
{templates_text}
""".strip()

# -----------------------------
# Task 3: OpenAI explanation
# -----------------------------
def generate_explanation_openai(row: pd.Series, model: str = DEFAULT_MODEL) -> str:
    # Convert row to dict of strings for safe API call
    row_dict = {k: str(v) for k, v in row.items()}
    reason_keys = generate_rule_based_reasons(row_dict)

    if not reason_keys:
        return "Transaction appears normal; no significant fraud indicators were detected."

    prompt = build_openai_prompt(row_dict, reason_keys)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional fraud detection analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# -----------------------------
# Batch generation
# -----------------------------
def generate_explanations_for_df_openai(df: pd.DataFrame, model: str = DEFAULT_MODEL) -> pd.DataFrame:
    df = df.copy()
    df["explanation"] = df.apply(
        lambda row: generate_explanation_openai(row, model=model),
        axis=1
    )
    return df

# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    import pandas as pd

    # Single-row test
    test_row = pd.DataFrame([{
        "transaction_id": "TX999",
        "transaction_amount": 12000,
        "merchant_category": "Electronics",
        "geo_mismatch": 1,
        "device_fingerprint_changed": True,
        "velocity_1h": 5,
        "avg_amount_30d": 2000,
        "fraud_score": 0.85,
        "fraud_prediction": 1,
        "transaction_timestamp": "2025-01-01 12:00:00"
    }])

    print("Generating explanation for single test transaction...")
    result_df = generate_explanations_for_df_openai(test_row)
    print(result_df[["transaction_id", "explanation"]].iloc[0])
