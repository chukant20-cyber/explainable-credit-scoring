import pandas as pd
import numpy as np
import pickle
import os
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def evaluate_fairness_uci_credit_card(processed_data_path, model_path, protected_attribute_names, privileged_groups, unprivileged_groups):
    print("Starting fairness evaluation for UCI Credit Card dataset...")

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

    # Create AIF360 dataset
    # Ensure protected attributes are in the DataFrame
    # For UCI, 'SEX_2' (Female) vs 'SEX_1' (Male) and 'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4' vs 'EDUCATION_1'
    # Let's assume 'SEX_2' (Female) is the unprivileged group for demonstration
    # And 'EDUCATION_1' (Graduate School) is privileged, others unprivileged

    # AIF360 requires the protected attributes to be present in the dataset
    # We need to reconstruct the original features or ensure the one-hot encoded columns are present
    # For simplicity, let's assume 'SEX_2' is our protected attribute for now
    # And 'TARGET' is the label column

    # Ensure all necessary columns are present for BinaryLabelDataset
    # This might require re-running preprocessing to keep original categorical columns or specific one-hot encoded ones
    # For now, let's simplify and assume 'SEX_2' is a direct column in X

    # Reconstruct df for AIF360, including original features if possible, or relevant one-hot encoded ones
    # For simplicity, let's use the processed dataframe and assume 'SEX_2' is the column for gender
    # This requires that 'SEX_2' was not dropped and is present in X

    # Let's check if 'SEX_2' is in X.columns
    if 'SEX_2' not in X.columns:
        print("Error: 'SEX_2' (Female) column not found in processed data. Cannot perform fairness evaluation based on gender.")
        return

    # Combine X and y for AIF360
    df_aif = X.copy()
    df_aif['TARGET'] = y
    df_aif['prediction'] = y_pred
    df_aif['probability'] = y_proba

    # Define privileged and unprivileged groups for 'SEX_2' (Female)
    # Assuming 'SEX_2' == 1 for Female (unprivileged), 'SEX_2' == 0 for Male (privileged)
    # This depends on how get_dummies encoded it. If drop_first=True, SEX_1 (Male) is dropped, SEX_2 (Female) remains.
    # So, SEX_2=1 is Female, SEX_2=0 is Male.
    # Let's assume Female is unprivileged (SEX_2=1)
    # And Male is privileged (SEX_2=0)

    # For AIF360, protected attributes are usually specified as lists of dictionaries
    # e.g., [{'SEX': 1}] for unprivileged group where SEX=1
    # We need to map our one-hot encoded columns back to a conceptual protected attribute
    # Or use the one-hot encoded columns directly as protected attributes

    # Let's use 'SEX_2' directly as a protected attribute for simplicity
    # privileged_groups = [{'SEX_2': 0}]
    # unprivileged_groups = [{'SEX_2': 1}]

    # Create BinaryLabelDataset
    bld = BinaryLabelDataset(df=df_aif,
                             label_names=['TARGET'],
                             protected_attribute_names=protected_attribute_names,
                             favorable_label=0, # Assuming 0 is non-default (favorable)
                             unfavorable_label=1)

    # Apply predictions to the dataset
    bld_pred = bld.copy()
    bld_pred.labels = df_aif['prediction'].values.reshape(-1, 1)

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

    print("Fairness evaluation complete for UCI Credit Card dataset.")

if __name__ == "__main__":
    processed_data_path = "explainable-credit-scoring/data_preprocessing/uci_credit_card_processed_data_fold0.csv"
    model_path = "explainable-credit-scoring/model_training/uci_credit_card_model.pkl"

    # Define protected attributes and groups for 'SEX_2' (Female vs Male)
    protected_attribute_names_sex = ['SEX_2']
    privileged_groups_sex = [{'SEX_2': 0}] # Male
    unprivileged_groups_sex = [{'SEX_2': 1}] # Female

    evaluate_fairness_uci_credit_card(processed_data_path, model_path, protected_attribute_names_sex, privileged_groups_sex, unprivileged_groups_sex)

    # Define protected attributes and groups for 'EDUCATION_1' (Graduate School vs Others)
    # This assumes 'EDUCATION_1' is present and represents Graduate School (privileged)
    # And other education levels (e.g., EDUCATION_2, EDUCATION_3, EDUCATION_4) are unprivileged
    # This would require more complex group definition or a different approach if only one-hot encoded columns are present
    # For simplicity, let's assume we can define groups based on the presence of specific one-hot columns
    # This part is conceptual and might need adjustment based on actual data structure after one-hot encoding
    # For now, we'll stick to SEX_2 as the primary example.




    with open("explainable-credit-scoring/fairness_evaluation/uci_credit_card_fairness_results.txt", "w") as f:
        f.write(f"Fairness metrics for protected attribute: {protected_attribute_names}\n")
        f.write(f"Mean difference in label (unprivileged - privileged): {metric_orig.mean_difference()}\n")
        f.write(f"Statistical Parity Difference: {metric_pred.statistical_parity_difference()}\n")
        f.write(f"Equal Opportunity Difference (FPR): {metric_pred.equal_opportunity_difference()}\n")
        f.write(f"Average Odds Difference: {metric_pred.average_odds_difference()}\n")
