import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def create_sample_fraud_csv(output_path="data/raw/fraud_model_output.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    random.seed(42)

    transactions = []
    for i in range(20):
        transaction_id = f"TX{i+1:03d}"
        transaction_amount = round(np.random.uniform(10, 15000), 2)
        merchant_category = random.choice(["Electronics", "Grocery", "Travel", "Clothing", "Restaurants"])
        
        # Required columns
        transaction_country = random.choice(["US", "AE", "IN", "GB"])
        customer_country = random.choice(["US", "AE", "IN", "GB"])
        geo_mismatch = 1 if transaction_country != customer_country else 0
        device_fingerprint_changed = random.choices([False, True], weights=[0.7, 0.3])[0]
        velocity_1h = np.random.randint(1, 6)
        high_velocity_flag = 1 if velocity_1h > 3 else 0
        avg_amount_30d = round(np.random.uniform(50, 5000), 2)
        fraud_score = round(np.random.uniform(0,1),2)
        fraud_prediction = 1 if i < 5 else (0 if i < 10 else random.choice([0,1]))
        transaction_timestamp = (datetime.now() - timedelta(days=random.randint(0,30))).strftime("%Y-%m-%d %H:%M:%S")
        
        # Synthetic / derived feature
        synthetic_feature = round(transaction_amount / (avg_amount_30d + 1),2)

        transactions.append({
            "transaction_id": transaction_id,
            "transaction_amount": transaction_amount,
            "merchant_category": merchant_category,
            "transaction_country": transaction_country,
            "customer_country": customer_country,
            "geo_mismatch": geo_mismatch,
            "device_fingerprint_changed": device_fingerprint_changed,
            "velocity_1h": velocity_1h,
            "high_velocity_flag": high_velocity_flag,
            "avg_amount_30d": avg_amount_30d,
            "fraud_score": fraud_score,
            "fraud_prediction": fraud_prediction,
            "transaction_timestamp": transaction_timestamp,
            "synthetic_feature": synthetic_feature
        })

    df = pd.DataFrame(transactions)
    df.to_csv(output_path, index=False)
    print(f"Sample fraud CSV created at {output_path}")


if __name__=="__main__":
    create_sample_fraud_csv()
