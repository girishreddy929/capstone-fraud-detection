import os
import pandas as pd
from openai import OpenAI
from src.explanation.templates import ExplanationTemplates
from src.config import OPENAI_API_KEY, DEFAULT_MODEL

# -----------------------------
# Initialize OpenAI client
# -----------------------------
# This sets up the OpenAI client with your API key.
# It first tries to get it from config, then from environment variable.
client = OpenAI(
    api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
)

# Ensure the API key is set. Otherwise, raise an error immediately.
if not client.api_key:
    raise ValueError(
        "OpenAI API key not set. Define OPENAI_API_KEY in src/config.py or environment."
    )

# -----------------------------
# Task 2: Rule-based reasons
# -----------------------------
def generate_rule_based_reasons(row: dict) -> list:
    """
    Identify which rule-based fraud reasons apply for a single transaction.

    Args:
        row (dict): Dictionary of transaction features

    Returns:
        list: List of reason keys that match
    """
    reasons = []

    try:
        # High transaction amount > 5000
        if float(row.get("transaction_amount", 0)) > 5000:
            reasons.append("high_transaction_amount")

        # Geo mismatch indicator
        if int(row.get("geo_mismatch", 0)) == 1:
            reasons.append("geo_mismatch")

        # Device fingerprint changed
        if bool(row.get("device_fingerprint_changed", False)):
            reasons.append("device_fingerprint_changed")

        # High velocity (multiple transactions in short time)
        if int(row.get("high_velocity_flag", 0)) == 1:
            reasons.append("high_velocity_flag")

        # High fraud score threshold
        if float(row.get("fraud_score", 0)) >= 0.8:
            reasons.append("high_fraud_score")

        # Large transaction + geo mismatch
        if float(row.get("transaction_amount", 0)) > 10000 and int(row.get("geo_mismatch", 0)) == 1:
            reasons.append("large_amount_geo_mismatch")

    except Exception as e:
        # Catch and print any errors evaluating the rules
        print(f"Error evaluating rule-based reasons: {e}")

    return reasons

# -----------------------------
# Prompt construction
# -----------------------------
def build_openai_prompt(row: dict, reason_keys: list) -> str:
    """
    Build the prompt to send to OpenAI based on the transaction row
    and detected rule-based reasons.

    Args:
        row (dict): Transaction data as dict of strings
        reason_keys (list): Rule-based reason keys

    Returns:
        str: Full prompt text
    """
    # Convert reason keys to human-readable text using templates
    templates_text = " ".join([ExplanationTemplates.get_template(k) for k in reason_keys])
    # Convert all values to string to prevent serialization/API errors
    row_str_dict = {k: str(v) for k, v in row.items()}

    # Build prompt string
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
    """
    Generate explanation for a single transaction using OpenAI.

    Args:
        row (pd.Series): Single transaction row
        model (str): OpenAI model to use

    Returns:
        str: Generated explanation text
    """
    # Convert pandas row to dictionary with string values for safe API call
    row_dict = {k: str(v) for k, v in row.items()}

    # Get rule-based reasons for this row
    reason_keys = generate_rule_based_reasons(row_dict)

    # If no reasons, return a default "normal" explanation
    if not reason_keys:
        return "Transaction appears normal; no significant fraud indicators were detected."

    # Build the prompt for OpenAI
    prompt = build_openai_prompt(row_dict, reason_keys)

    try:
        # Call OpenAI Chat Completions API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional fraud detection analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150,
        )

        # Return the generated explanation
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Catch API errors and return as string
        return f"Error generating explanation: {str(e)}"

# -----------------------------
# Batch generation
# -----------------------------
def generate_explanations_for_df_openai(df: pd.DataFrame, model: str = DEFAULT_MODEL) -> pd.DataFrame:
    """
    Generate explanations for all rows in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame of transactions
        model (str): OpenAI model

    Returns:
        pd.DataFrame: DataFrame with a new 'explanation' column
    """
    df = df.copy()
    # Apply the single-row explanation function to each row
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

    # Load the real dataset
    csv_path = "data/raw/fraud_model_output.csv"
    df = pd.read_csv(csv_path)

    # Take only the first row for testing
    test_row = df.head(2)

    print("Generating explanation for first transaction in dataset...")
    result_df = generate_explanations_for_df_openai(test_row)

    # Print only transaction ID and explanation
    print(result_df)

