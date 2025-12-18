class ExplanationTemplates:
    """
    Comprehensive rule-based templates for common fraud patterns
    based on our fraud model output dataset.
    Includes placeholders for dynamic feature insertion.
    """

    TEMPLATES = {
        # Transaction amount patterns
        "high_transaction_amount": {
            "template": "Transaction amount of ${transaction_amount} is unusually high compared to typical customer behavior.",
            "category": "amount",
            "severity": "medium"
        },
        "very_high_transaction_amount": {
            "template": "Transaction amount of ${transaction_amount} is extremely high relative to customer's historical spending.",
            "category": "amount",
            "severity": "high"
        },

        # Location patterns
        "location_mismatch": {
            "template": "Transaction occurred in {transaction_country}, far from the customer's usual location ({customer_country}).",
            "category": "location",
            "severity": "medium"
        },
        "geo_mismatch": {
            "template": "Transaction country ({transaction_country}) does not match customer country ({customer_country}).",
            "category": "location",
            "severity": "high"
        },

        # Device patterns
        "new_device": {
            "template": "Transaction was performed on an unrecognized device.",
            "category": "device",
            "severity": "medium"
        },
        "device_fingerprint_changed": {
            "template": "Device fingerprint has changed since the last known transaction.",
            "category": "device",
            "severity": "medium"
        },

        # Velocity / frequency patterns
        "velocity_trigger": {
            "template": "Multiple rapid transactions occurred in a short time period.",
            "category": "velocity",
            "severity": "medium"
        },
        "high_velocity_flag": {
            "template": "Transaction frequency in a short period is unusually high.",
            "category": "velocity",
            "severity": "high"
        },

        # Fraud score patterns
        "high_fraud_score": {
            "template": "The fraud model assigned a high risk score ({fraud_score}) to this transaction.",
            "category": "score",
            "severity": "high"
        },

        # Historical behavior deviations
        "avg_amount_deviation": {
            "template": "Transaction amount of ${transaction_amount} deviates significantly from customer's average transaction amount (${avg_amount_30d}).",
            "category": "historical",
            "severity": "medium"
        },
        "merchant_category_anomaly": {
            "template": "Transaction occurred in an unusual merchant category ({merchant_category}) for this customer.",
            "category": "historical",
            "severity": "medium"
        },
        "time_of_day_anomaly": {
            "template": "Transaction occurred at an unusual time of day ({transaction_timestamp}).",
            "category": "historical",
            "severity": "medium"
        },

        # Combined / complex patterns
        "large_amount_geo_mismatch": {
            "template": "A high-value transaction (${transaction_amount}) occurred in a country ({transaction_country}) different from the customer's usual location ({customer_country}).",
            "category": "complex",
            "severity": "high"
        },
        "high_velocity_new_device": {
            "template": "Multiple rapid transactions were made from a new device.",
            "category": "complex",
            "severity": "high"
        },
        "complex_fraud_pattern": {
            "template": "This transaction exhibits multiple risk factors including high amount, location mismatch, and device change.",
            "category": "complex",
            "severity": "high"
        }
    }

    @staticmethod
    def get_template(reason_key: str, **kwargs) -> str:
        """
        Retrieve a template by key and optionally format it with provided keyword arguments.
        """
        entry = ExplanationTemplates.TEMPLATES.get(reason_key)
        if entry is None:
            return "No template available for this reason."
        template_text = entry["template"]
        try:
            return template_text.format(**kwargs)
        except KeyError:
            # If placeholders not provided, return unformatted template
            return template_text

    @staticmethod
    def list_all_templates() -> dict:
        """
        Returns all templates as a dictionary.
        """
        return ExplanationTemplates.TEMPLATES
