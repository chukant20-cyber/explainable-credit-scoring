import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats
import statsmodels.stats.multitest as smm

def statistical_testing_uci_credit_card(shap_values_path, processed_data_path, random_state=42):
    print("Starting statistical testing for UCI Credit Card dataset...")

    if not os.path.exists(shap_values_path):
        print(f"Error: SHAP values not found at {shap_values_path}. Please run explanation generation first.")
        return
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data not found at {processed_data_path}. Please run data preprocessing first.")
        return

    with open(shap_values_path, "rb") as f:
        shap_data = pickle.load(f)
    shap_values = shap_data["shap_values"]
    X_sample = shap_data["X_sample"]

    # Ensure shap_values_array is numerical
    if isinstance(shap_values, list):
        shap_values_array = np.array(shap_values[1], dtype=float) # Assuming positive class explanations
    else:
        shap_values_array = np.array(shap_values, dtype=float)

    # Perform t-test for each feature: Is the mean SHAP value significantly different from zero?
    print("\n--- T-tests for mean SHAP values (vs. zero) ---")
    feature_p_values = {}
    for i, feature in enumerate(X_sample.columns):
        if shap_values_array.shape[0] > 1: # Need at least 2 samples for t-test
            t_stat, p_value = stats.ttest_1samp(shap_values_array[:, i], 0)
            feature_p_values[feature] = p_value
        else:
            feature_p_values[feature] = np.nan # Cannot perform t-test with single sample

    # Multiple comparison correction (e.g., Benjamini-Hochberg FDR correction)
    features = list(feature_p_values.keys())
    p_values = np.array(list(feature_p_values.values()))
    
    # Filter out NaN p-values before correction
    valid_p_values_mask = ~np.isnan(p_values)
    if np.any(valid_p_values_mask):
        rejected, p_values_corrected, _, _ = smm.multipletests(p_values[valid_p_values_mask], alpha=0.05, method='fdr_bh')
        
        corrected_results = {}
        j = 0
        for i, feature in enumerate(features):
            if valid_p_values_mask[i]:
                corrected_results[feature] = {"p_value_corrected": p_values_corrected[j], "rejected": rejected[j]}
                j += 1

        print("\n--- Statistical Significance after FDR Correction (alpha=0.05) ---")
        for feature, res in corrected_results.items():
            if res["rejected"]:
                print(f"Feature {feature}: Corrected p={res['p_value_corrected']:.4f} (Significant)")
            # else:
            #     print(f"Feature {feature}: Corrected p={res["p_value_corrected"]:.4f} (Not Significant)")
    else:
        print("No valid p-values to perform multiple comparison correction.")

    print("Statistical testing complete for UCI Credit Card dataset.")

if __name__ == "__main__":
    processed_data_path = "explainable-credit-scoring/data_preprocessing/uci_credit_card_processed_data_fold0.csv"
    shap_values_path = "explainable-credit-scoring/explanation_generation/uci_credit_card_shap_values.pkl"
    statistical_testing_uci_credit_card(shap_values_path, processed_data_path)

