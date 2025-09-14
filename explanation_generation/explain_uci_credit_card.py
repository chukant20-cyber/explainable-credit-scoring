import pandas as pd
import numpy as np
import pickle
import shap
import os

def explain_uci_credit_card_model(processed_data_path, model_path, random_state=42):
    print("Starting explanation generation for UCI Credit Card dataset...")

    if not os.path.exists(processed_data_path):
        print(f"Error: Preprocessed data not found at {processed_data_path}. Please run data preprocessing first.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Please run model training first.")
        return

    df_processed = pd.read_csv(processed_data_path)
    
    X_test = df_processed.drop(columns=["TARGET"])
    y_test = df_processed["TARGET"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Select a subset of data for explanation to manage computation time
    # For stability analysis, we might select multiple instances or a small random sample
    np.random.seed(random_state)
    sample_indices = np.random.choice(X_test.index, min(100, len(X_test)), replace=False)
    X_sample = X_test.loc[sample_indices]

    # SHAP Explainer
    # For tree-based models, TreeExplainer is more efficient. For Logistic Regression, KernelExplainer or LinearExplainer.
    # Since we used Logistic Regression, we'll use LinearExplainer if available, otherwise KernelExplainer.
    try:
        explainer = shap.LinearExplainer(model, X_test)
    except Exception:
        print("LinearExplainer not applicable, falling back to KernelExplainer. This might be slower.")
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))

    shap_values = explainer.shap_values(X_sample)

    print("SHAP explanation generated for UCI Credit Card dataset.")
    print(f"Shape of SHAP values: {shap_values[0].shape if isinstance(shap_values, list) else shap_values.shape}")

    # Example of stability analysis (conceptual - needs more robust implementation)
    # One way to assess stability is to see how explanations change for similar instances
    # or for the same instance with slight perturbations.
    # For this example, we'll just print the mean absolute SHAP value for each feature.
    if isinstance(shap_values, list): # For multi-output models like LogisticRegression.predict_proba
        mean_abs_shap = np.mean(np.abs(shap_values[1]), axis=0) # Explanations for positive class
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    feature_importance_shap = pd.Series(mean_abs_shap, index=X_test.columns).astype(float)
    print("\nMean Absolute SHAP values (Top 10 features):\n", feature_importance_shap.nlargest(10))

    # Save explanations to a text file for testing
    with open("explainable-credit-scoring/explanation_generation/uci_credit_card_explanations.txt", "w") as f:
        f.write("UCI Credit Card Explanations\n")
        f.write(str(feature_importance_shap.nlargest(10)))

    # Save SHAP values for further analysis
    shap_output_path = "explainable-credit-scoring/explanation_generation/uci_credit_card_shap_values.pkl"
    with open(shap_output_path, "wb") as f:
        pickle.dump({"shap_values": shap_values, "X_sample": X_sample}, f)
    print(f"SHAP values saved to {shap_output_path}")

if __name__ == "__main__":
    processed_data_path = "explainable-credit-scoring/data_preprocessing/uci_credit_card_processed_data_fold0.csv"
    model_path = "explainable-credit-scoring/model_training/uci_credit_card_model.pkl"
    explain_uci_credit_card_model(processed_data_path, model_path)


