import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os
import pickle

def preprocess_lending_club(file_path, random_state=42):
    """
    Preprocesses the LendingClub dataset (loans_full_schema.csv) according to the specified protocol.
    """
    df = pd.read_csv(file_path)

    # Define target variable (loan_status) and map to binary (1 for default/charged-off, 0 otherwise)
    # Based on the description: "Target definition: Loan status charged-off or default"
    df["TARGET"] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)
    df = df.drop("loan_status", axis=1)

    # Drop columns that are not features or leak information
    # 'emp_title', 'issue_month', 'sub_grade', 'grade' (derived from interest_rate), 'initial_listing_status', 'disbursement_method'
    # 'paid_total', 'paid_principal', 'paid_interest', 'paid_late_fees' are post-loan outcome, 'balance' is current balance
    # 'annual_income_joint', 'verification_income_joint', 'debt_to_income_joint' have many NAs and might be redundant if individual income is present
    # 'months_since_last_delinq', 'months_since_90d_late', 'months_since_last_credit_inquiry' have many NAs and might be too specific
    # 'emp_length' needs to be converted to numerical

    # Columns to drop based on description and initial inspection
    cols_to_drop = [
        'emp_title', 'issue_month', 'sub_grade', 'grade', 'initial_listing_status', 'disbursement_method',
        'paid_total', 'paid_principal', 'paid_interest', 'paid_late_fees', 'balance',
        'annual_income_joint', 'verification_income_joint', 'debt_to_income_joint' # High missingness / joint info
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Convert emp_length to numerical
    df['emp_length'] = df['emp_length'].replace({'< 1 year': '0', '10+ years': '10', 'years': '', 'year': ''}, regex=True).astype(float)

    # Earliest credit line to age of credit
    df['earliest_credit_line'] = pd.to_datetime(df['earliest_credit_line'], format='%Y').dt.year
    df['credit_line_age'] = 2025 - df['earliest_credit_line'] # Assuming current year is 2025 as per prompt
    df = df.drop('earliest_credit_line', axis=1)

    # Missing Value Treatment:
    # Continuous variables: Median imputation within training folds only (handled by pipeline)
    # Categorical variables: Explicit "missing" category preserved as informative signal (handled by one-hot encoding)
    # High-missingness features (>80% missing): Removed from analysis
    # Let's check missingness
    missing_percentages = df.isnull().sum() / len(df) * 100
    high_missingness_cols = missing_percentages[missing_percentages > 80].index.tolist()
    df = df.drop(columns=high_missingness_cols, errors='ignore')

    # Identify categorical and numerical features for preprocessing
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    numerical_features.remove('TARGET')

    # Preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # Ensure consistent feature names and dense output
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X = df.drop('TARGET', axis=1)
    y = df['TARGET']

    # Data Splitting Protocol: 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit preprocessor on training data and transform both train and test
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Get feature names after one-hot encoding
        ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        all_feature_names = numerical_features + ohe_feature_names.tolist()
        print(f"DEBUG: All feature names for fold {fold_idx}: {all_feature_names}")

        X_train_processed = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)

        folds.append({
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test
        })

    print(f"Preprocessing complete for LendingClub dataset. Generated {len(folds)} stratified folds.")
    return folds, all_feature_names

if __name__ == '__main__':
    lending_club_path = '/home/ubuntu/upload/loans_full_schema.csv'
    folds, feature_names = preprocess_lending_club(lending_club_path)

    print(f"Number of features after preprocessing: {len(feature_names)}")
    print(f"Shape of first fold X_train: {folds[0]['X_train'].shape}")
    print(f"Shape of first fold y_train: {folds[0]['y_train'].shape}")
    print(f"Shape of first fold X_test: {folds[0]['X_test'].shape}")
    print(f"Shape of first fold y_test: {folds[0]['y_test'].shape}")

    # Save preprocessed data for the first fold as an example
    processed_df_path = "explainable-credit-scoring/data_preprocessing/lending_club_processed_data_fold0.csv"
    os.makedirs(os.path.dirname(processed_df_path), exist_ok=True)
    processed_train_df = folds[0]["X_train"].copy()
    processed_train_df["TARGET"] = folds[0]["y_train"]
    processed_train_df.to_csv(processed_df_path, index=False)
    print(f"First fold preprocessed data saved to {processed_df_path}")

    folds_info_path = "explainable-credit-scoring/data_preprocessing/lending_club_folds_info.pkl"
    with open(folds_info_path, 'wb') as f:
        pickle.dump([{'train_idx': fold['X_train'].index.tolist(), 'test_idx': fold['X_test'].index.tolist()} for fold in folds], f)
    print(f"Folds indices saved to {folds_info_path}")

