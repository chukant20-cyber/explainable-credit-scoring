import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import os

def train_lending_club_model(processed_data_path, random_state=42):
    print("Starting model training for LendingClub dataset...")

    # Load preprocessed data for the first fold (as an example)
    df_processed = pd.read_csv(processed_data_path)
    
    X_train = df_processed.drop(columns=["TARGET"])
    y_train = df_processed["TARGET"]

    # Hyperparameter specifications
    model = LogisticRegression(
        solver='liblinear',
        random_state=random_state,
        C=0.5, # Regularization strength
        penalty='l1'
    )

    model.fit(X_train, y_train)

    # Evaluate on training data
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]

    accuracy = accuracy_score(y_train, y_pred)
    roc_auc = roc_auc_score(y_train, y_proba)

    print(f"Model training complete for LendingClub dataset.")
    print(f"Training Accuracy: {accuracy:.4f}")
    print(f"Training ROC AUC: {roc_auc:.4f}")

    # Save the trained model
    model_save_path = "explainable-credit-scoring/model_training/lending_club_model.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {model_save_path}")

    return model, accuracy, roc_auc

if __name__ == "__main__":
    processed_data_path = "explainable-credit-scoring/data_preprocessing/lending_club_processed_data_fold0.csv"
    if os.path.exists(processed_data_path):
        trained_model, acc, auc = train_lending_club_model(processed_data_path)
    else:
        print(f"Error: Preprocessed data not found at {processed_data_path}. Please run data preprocessing first.")

