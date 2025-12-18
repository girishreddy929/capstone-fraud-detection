from pydantic import BaseModel, validator
from datetime import datetime

class FraudModelOutput(BaseModel):
    transaction_id: str
    transaction_amount: float
    merchant_category: str
    transaction_country: str
    customer_country: str
    device_fingerprint_changed: bool
    transaction_timestamp: datetime
    velocity_1h: int
    avg_amount_30d: float
    fraud_score: float
    fraud_prediction: int  # 0 = Not Fraud, 1 = Fraud

    @validator("fraud_prediction")
    def fraud_label_must_be_0_or_1(cls, v):
        if v not in (0, 1):
            raise ValueError("fraud_prediction must be 0 or 1")
        return v

    @validator("transaction_timestamp", pre=True)
    def parse_transaction_timestamp(cls, v):
        # Ensure datetime parsing works for string timestamps
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            except ValueError as e:
                raise ValueError(f"Invalid transaction_timestamp format: {v}") from e
        return v
