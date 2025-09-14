import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
import pickle

def preprocess_uci_credit_card(file_path, random_state=42):
    """
    Preprocesses the UCI Credit Card dataset according to the specified protocol.
    """
    df = pd.read_csv(file_path)

    # Rename target column for consistency
    df = df.rename(columns={'default.payment.next.month': 'TARGET'})

    # Drop ID column as it's not a feature
    df = df.drop('ID', axis=1)

    # No missing value treatment or outlier handling needed as per document for this dataset
    # "Data quality: No missing values, pre-processed by original authors"

    # Feature Engineering: Categorical encoding (one-hot for low cardinality, target for higher)
    # For this dataset, most categorical features are already numerical or low cardinality.
    # SEX, EDUCATION, MARRIAGE are categorical. AGE is numerical.
    # PAY_0 to PAY_6 are ordinal/categorical.
    # We'll treat SEX, EDUCATION, MARRIAGE as categorical and one-hot encode them.
    # PAY_X features are payment status, which can be treated as ordinal or categorical.
    # Given the context, we'll treat them as categorical for one-hot encoding for simplicity and robustness.

    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Feature scaling: StandardScaler (only for neural network models, but we'll include it as a step)
    # We'll scale all non-binary features (excluding the target and one-hot encoded columns).
    # Identify numerical columns that are not one-hot encoded or the target
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['TARGET'] and not col.startswith(('SEX_', 'EDUCATION_', 'MARRIAGE_', 'PAY_'))]

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Data Splitting Protocol: 5-fold stratified CV
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        folds.append({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test})

    print(f"Preprocessing complete for UCI Credit Card dataset. Generated {len(folds)} stratified folds.")
    return folds, df.columns.tolist()

if __name__ == '__main__':
    # Example usage:
    uci_credit_card_path = '/home/ubuntu/upload/UCI_Credit_Card.csv'
    folds, feature_names = preprocess_uci_credit_card(uci_credit_card_path)

    print(f"Number of features after preprocessing: {len(feature_names) - 1}") # -1 for target
    print(f"Shape of first fold X_train: {folds[0]['X_train'].shape}")
    print(f"Shape of first fold y_train: {folds[0]['y_train'].shape}")
    print(f"Shape of first fold X_test: {folds[0]['X_test'].shape}")
    print(f"Shape of first fold y_test: {folds[0]['y_test'].shape}")

    # Save preprocessed data for the first fold as an example
    processed_df_path = "explainable-credit-scoring/data_preprocessing/uci_credit_card_processed_data_fold0.csv"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(processed_df_path), exist_ok=True)
    
    # Combine X_train and y_train for saving
    processed_train_df = folds[0]["X_train"].copy()
    processed_train_df["TARGET"] = folds[0]["y_train"]
    processed_train_df.to_csv(processed_df_path, index=False)
    print(f"First fold preprocessed data saved to {processed_df_path}")

    folds_info_path = "explainable-credit-scoring/data_preprocessing/uci_credit_card_folds_info.pkl"
    with open(folds_info_path, "wb") as f:
        pickle.dump([{"train_idx": fold["X_train"].index.tolist(), "test_idx": fold["X_test"].index.tolist()} for fold in folds], f)
    print(f"Folds indices saved to {folds_info_path}")

