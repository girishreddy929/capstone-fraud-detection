import shap
import pandas as pd
import matplotlib.pyplot as plt

def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values for the given model and feature DataFrame.

    Args:
        model: Trained ML model (e.g., XGBoost, RandomForest, LightGBM)
        X (pd.DataFrame): Feature DataFrame used for predictions

    Returns:
        shap.Explanation: SHAP values object
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values


def plot_global_feature_importance(shap_values, X: pd.DataFrame, max_features: int = 10):
    """
    Plot global feature importance using mean absolute SHAP values.
    """
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_features)


def plot_local_waterfall(shap_values, transaction_index: int):
    """
    Plot waterfall for a single transaction.
    """
    if transaction_index >= len(shap_values):
        raise IndexError("transaction_index out of range for SHAP values")
    shap.plots.waterfall(shap_values[transaction_index])


def get_top_features_per_transaction(shap_values, X: pd.DataFrame, transaction_index: int, top_n: int = 5):
    """
    Return top N contributing features for a transaction.
    """
    if transaction_index >= len(shap_values):
        raise IndexError("transaction_index out of range for SHAP values")

    sv = shap_values.values[transaction_index]
    feature_names = X.columns
    feature_contributions = list(zip(feature_names, sv))
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return feature_contributions[:top_n]


def get_top_features_df(shap_values, X: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Return a DataFrame of top contributing features for each transaction.
    """
    all_top_features = []

    for idx in range(len(shap_values)):
        top_features = get_top_features_per_transaction(shap_values, X, idx, top_n)
        row = {"transaction_index": idx}
        for i, (feature, value) in enumerate(top_features, start=1):
            row[f"top_feature_{i}"] = feature
            row[f"top_feature_value_{i}"] = value
        all_top_features.append(row)

    return pd.DataFrame(all_top_features)
