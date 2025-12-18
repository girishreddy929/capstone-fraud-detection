import pandas as pd
from src.data_loader.schema import FraudModelOutput
from typing import List

# Original required columns
REQUIRED_COLUMNS = [
    "transaction_id",
    "transaction_amount",
    "merchant_category",
    "transaction_country",
    "customer_country",
    "device_fingerprint_changed",
    "transaction_timestamp",
    "velocity_1h",
    "avg_amount_30d",
    "fraud_score",
    "fraud_prediction",
]

# Derived/synthetic features to validate
DERIVED_COLUMNS = ["geo_mismatch", "high_velocity_flag"]

ALL_COLUMNS = REQUIRED_COLUMNS + DERIVED_COLUMNS


def load_and_validate_fraud_output(csv_path: str) -> pd.DataFrame:
    """
    Load a fraud model output CSV, validate required and derived columns,
    and return a validated DataFrame including derived features.
    """
    df = pd.read_csv(csv_path)

    # 1️⃣ Validate column existence
    missing_columns = set(ALL_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required/derived columns: {missing_columns}")

    # 2️⃣ Check for null values
    null_columns = df[ALL_COLUMNS].columns[df[ALL_COLUMNS].isnull().any()]
    if len(null_columns) > 0:
        raise ValueError(f"Null values found in columns: {list(null_columns)}")

    # 3️⃣ Schema validation for required columns
    validated_records: List[dict] = []
    for idx, row in df.iterrows():
        try:
            # Validate required fields using Pydantic schema
            required_data = {k: row[k] for k in REQUIRED_COLUMNS}
            validated = FraudModelOutput(**required_data)

            # Add derived columns as-is
            for derived_col in DERIVED_COLUMNS:
                validated.__dict__[derived_col] = row[derived_col]

            validated_records.append(validated.__dict__)
        except Exception as e:
            raise ValueError(f"Row {idx} failed validation: {e}")

    validated_df = pd.DataFrame(validated_records)

    print(f"Successfully validated {len(validated_df)} records.")
    return validated_df


# Example usage
if __name__ == "__main__":
    df_validated = load_and_validate_fraud_output("data/raw/fraud_model_output.csv")
    print(df_validated.head())
