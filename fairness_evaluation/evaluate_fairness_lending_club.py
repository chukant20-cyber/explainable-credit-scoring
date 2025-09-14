import pandas as pd
import numpy as np
import pickle
import os
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def evaluate_fairness_lending_club(processed_data_path, model_path, protected_attribute_names, privileged_groups, unprivileged_groups):
    print("Starting fairness evaluation for LendingClub dataset...")

    if not os.path.exists(processed_data_path):
        print(f"Error: Preprocessed data not found at {processed_data_path}. Please run data preprocessing first.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Please run model training first.")
        return

    df_processed = pd.read_csv(processed_data_path)
    
    X = df_processed.drop(columns=["TARGET"])
    y = df_processed["TARGET"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Combine X and y for AIF360
    df_aif = X.copy()
    df_aif["TARGET"] = y
    df_aif["prediction"] = y_pred
    df_aif["probability"] = y_proba

    # Ensure protected attributes are present in the DataFrame
    for attr in protected_attribute_names:
        if attr not in df_aif.columns:
            print(f"Error: Protected attribute \'{attr}\' not found in processed data. Cannot perform fairness evaluation.")
            return

    # Create BinaryLabelDataset
    bld = BinaryLabelDataset(df=df_aif,
                             label_names=["TARGET"],
                             protected_attribute_names=protected_attribute_names,
                             favorable_label=0, # Assuming 0 is non-default (favorable)
                             unfavorable_label=1)

    # Apply predictions to the dataset
    bld_pred = bld.copy()
    bld_pred.labels = df_aif["prediction"].values.reshape(-1, 1)

    # Calculate fairness metrics
    metric_orig = BinaryLabelDatasetMetric(bld, 
                                           privileged_groups=privileged_groups, 
                                           unprivileged_groups=unprivileged_groups)
    
    metric_pred = ClassificationMetric(bld, bld_pred, 
                                       unprivileged_groups=unprivileged_groups, 
                                       privileged_groups=privileged_groups)

    print(f"\nFairness metrics for protected attribute: {protected_attribute_names}")
    print(f"Mean difference in label (unprivileged - privileged): {metric_orig.mean_difference()}")
    print(f"Statistical Parity Difference: {metric_pred.statistical_parity_difference()}")
    print(f"Equal Opportunity Difference (FPR): {metric_pred.equal_opportunity_difference()}") # Difference in False Positive Rates
    print(f"Average Odds Difference: {metric_pred.average_odds_difference()}")

    print("Fairness evaluation complete for LendingClub dataset.")

if __name__ == "__main__":
    processed_data_path = "explainable-credit-scoring/data_preprocessing/lending_club_processed_data_fold0.csv"
    model_path = "explainable-credit-scoring/model_training/lending_club_model.pkl"

    # Define protected attributes and groups for 'person_home_ownership_Rent' (Rent vs Own/Mortgage)
    # Assuming 'person_home_ownership_Rent' is 1 for Rent (unprivileged) and 0 for others (privileged)
    protected_attribute_names_home_ownership = ["homeownership_RENT"]
    privileged_groups_home_ownership = [{'homeownership_RENT': 0}] # Own/Mortgage
    unprivileged_groups_home_ownership = [{'homeownership_RENT': 1}] # Rent

    evaluate_fairness_lending_club(processed_data_path, model_path, protected_attribute_names_home_ownership, privileged_groups_home_ownership, unprivileged_groups_home_ownership)



    with open("explainable-credit-scoring/fairness_evaluation/lending_club_fairness_results.txt", "w") as f:
        f.write(f"Fairness metrics for protected attribute: {protected_attribute_names}\n")
        f.write(f"Mean difference in label (unprivileged - privileged): {metric_orig.mean_difference()}\n")
        f.write(f"Statistical Parity Difference: {metric_pred.statistical_parity_difference()}\n")
        f.write(f"Equal Opportunity Difference (FPR): {metric_pred.equal_opportunity_difference()}\n")
        f.write(f"Average Odds Difference: {metric_pred.average_odds_difference()}\n")
