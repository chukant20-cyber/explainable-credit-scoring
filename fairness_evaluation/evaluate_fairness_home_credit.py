import pandas as pd
import numpy as np
import pickle
import os
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def evaluate_fairness_home_credit(processed_data_path, model_path, protected_attribute_names, privileged_groups, unprivileged_groups):
    print("Starting fairness evaluation for Home Credit Default Risk dataset...")

    if not os.path.exists(processed_data_path):
        print(f"Error: Preprocessed data not found at {processed_data_path}. Skipping fairness evaluation for Home Credit.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Skipping fairness evaluation for Home Credit.")
        return

    df_processed = pd.read_csv(processed_data_path)


    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X = df_processed.drop(columns=["TARGET"])
    y = df_processed["TARGET"]

    # Predict probabilities and labels
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Create AIF360 dataset
    # Assuming 'CODE_GENDER_F' is the protected attribute, 0 for male (privileged), 1 for female (unprivileged)
    # This needs to be dynamically set based on protected_attribute_names and groups
    # For simplicity, using CODE_GENDER_F as an example





    bld = BinaryLabelDataset(df=df_processed, label_names=["TARGET"],
                             protected_attribute_names=protected_attribute_names,
                             favorable_label=1, unfavorable_label=0,
                             privileged_protected_attributes=privileged_groups,
                             unprivileged_protected_attributes=unprivileged_groups) # Assuming 1 is favorable, 0 is unfavorable
    bld_pred = bld.copy()
    bld_pred.labels = y_pred.reshape(-1, 1)
    bld_pred.scores = y_pred_proba.reshape(-1, 1)

    # Calculate fairness metrics
    metric_orig = BinaryLabelDatasetMetric(bld, 
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)
    
    metric_pred = ClassificationMetric(bld, bld_pred, 
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

    print(f"Fairness metrics for protected attribute: {protected_attribute_names}")
    print(f"Mean difference in label (unprivileged - privileged): {metric_orig.mean_difference()}")
    print(f"Statistical Parity Difference: {metric_pred.statistical_parity_difference()}")
    print(f"Equal Opportunity Difference (FPR): {metric_pred.equal_opportunity_difference(privileged=False)}")
    print(f"Average Odds Difference: {metric_pred.average_odds_difference()}")
    print("Fairness evaluation complete for Home Credit Default Risk dataset.")

if __name__ == "__main__":
    processed_data_path = "explainable-credit-scoring/data_preprocessing/home_credit_processed_data_fold0.csv"
    model_path = "explainable-credit-scoring/model_training/home_credit_model.pkl"

    # Placeholder for protected attributes and groups
    protected_attribute_names_example = ["CODE_GENDER_F"]
    privileged_groups_example = [{'CODE_GENDER_F': 0}]
    unprivileged_groups_example = [{'CODE_GENDER_F': 1}]
    evaluate_fairness_home_credit(processed_data_path, model_path, protected_attribute_names_example, privileged_groups_example, unprivileged_groups_example)
n