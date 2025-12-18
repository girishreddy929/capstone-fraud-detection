# test_single_row_openai.py

import os
import pandas as pd
from openai import OpenAI
from src.config import OPENAI_API_KEY, DEFAULT_MODEL

MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
)
# Example single row (replace with your actual transaction data)
row = {
    "transaction_id": "TX001",
    "transaction_amount": 12000,
    "merchant_category": "Electronics",
    "geo_mismatch": 1,
    "device_fingerprint_changed": True,
    "velocity_1h": 5,
    "avg_amount_30d": 500,
    "fraud_score": 0.85,
    "fraud_prediction": 1,
    "transaction_timestamp": "2025-01-01 02:00:00"
}

# Build a minimal prompt
prompt = f"""
You are a fraud analyst assistant.

Explain why this transaction was flagged as potentially fraudulent.
Transaction features: {row}
"""

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a professional fraud detection analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )

    print("✅ Explanation generated successfully:")
    print(response.choices[0].message.content.strip())

except Exception as e:
    print("❌ API call failed")
    print(str(e))
