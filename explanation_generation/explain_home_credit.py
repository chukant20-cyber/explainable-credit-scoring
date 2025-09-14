import pandas as pd
import numpy as np
import pickle
import shap
import os

def explain_home_credit_model(processed_data_path, model_path, random_state=42):
    print("Starting explanation generation for Home Credit Default Risk dataset...")

    if not os.path.exists(processed_data_path):
        print(f"Error: Preprocessed data not found at {processed_data_path}. Skipping explanation generation for Home Credit.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Skipping explanation generation for Home Credit.")
        return

    try:
        df_processed = pd.read_csv(processed_data_path)
    except Exception as e:
        print(f"Error loading processed data for Home Credit: {e}")
        print("This might be due to memory limitations. Skipping explanation generation for Home Credit.")
        return
    
    X_test = df_processed.drop(columns=["TARGET"])
    y_test = df_processed["TARGET"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Select a subset of data for explanation to manage computation time
    np.random.seed(random_state)
    sample_indices = np.random.choice(X_test.index, min(100, len(X_test)), replace=False)
    X_sample = X_test.loc[sample_indices]

    # SHAP Explainer
    try:
        explainer = shap.LinearExplainer(model, X_test)
    except Exception:
        print("LinearExplainer not applicable, falling back to KernelExplainer. This might be slower.")
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))

    shap_values = explainer.shap_values(X_sample)

    print("SHAP explanation generated for Home Credit Default Risk dataset.")
    print(f"Shape of SHAP values: {shap_values[0].shape if isinstance(shap_values, list) else shap_values.shape}")

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean(np.abs(shap_values[1]), axis=0)
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    feature_importance_shap = pd.Series(mean_abs_shap, index=X_test.columns)
    print("\nMean Absolute SHAP values (Top 10 features):\n", feature_importance_shap.nlargest(10))

    # Save explanations to a text file for testing
    with open("explainable-credit-scoring/explanation_generation/home_credit_explanations.txt", "w") as f:
        f.write("Home Credit Explanations\n")
        f.write(str(feature_importance_shap.nlargest(10)))

    # Save SHAP values for further analysis
    shap_output_path = "explainable-credit-scoring/explanation_generation/home_credit_shap_values.pkl"
    with open(shap_output_path, "wb") as f:
        pickle.dump({"shap_values": shap_values, "X_sample": X_sample}, f)
    print(f"SHAP values saved to {shap_output_path}")

if __name__ == "__main__":
    processed_data_path = "explainable-credit-scoring/data_preprocessing/home_credit_processed_data_fold0.csv"
    model_path = "explainable-credit-scoring/model_training/home_credit_model.pkl"
    explain_home_credit_model(processed_data_path, model_path)

