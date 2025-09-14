import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import os

def train_home_credit_model(processed_data_path, random_state=42):
    print("Starting model training for Home Credit Default Risk dataset...")

    # Load preprocessed data for the first fold (as an example)
    # Note: Due to memory constraints, this script assumes that the preprocessed data
    # for a single fold can be loaded. For full cross-validation or larger datasets,
    # more advanced memory management (e.g., Dask, PySpark) would be required.
    try:
        df_processed = pd.read_csv(processed_data_path)
    except Exception as e:
        print(f"Error loading processed data for Home Credit: {e}")
        print("This might be due to memory limitations. Skipping model training for Home Credit.")
        return None, None, None
    
    X_train = df_processed.drop(columns=["TARGET"])
    y_train = df_processed["TARGET"]

    # Hyperparameter specifications
    model = LogisticRegression(
        solver='liblinear',
        random_state=random_state,
        C=0.1, # Regularization strength, adjusted for potentially large feature set
        penalty='l1'
    )

    model.fit(X_train, y_train)

    # Evaluate on training data
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]

    accuracy = accuracy_score(y_train, y_pred)
    roc_auc = roc_auc_score(y_train, y_proba)

    print(f"Model training complete for Home Credit Default Risk dataset.")
    print(f"Training Accuracy: {accuracy:.4f}")
    print(f"Training ROC AUC: {roc_auc:.4f}")

    # Save the trained model
    model_save_path = "explainable-credit-scoring/model_training/home_credit_model.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {model_save_path}")

    return model, accuracy, roc_auc

if __name__ == "__main__":
    processed_data_path = "explainable-credit-scoring/data_preprocessing/home_credit_processed_data_fold0.csv"
    if os.path.exists(processed_data_path):
        trained_model, acc, auc = train_home_credit_model(processed_data_path)
    else:
        print(f"Error: Preprocessed data not found at {processed_data_path}. Please run data preprocessing first.")

