
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import gc
import os
import tempfile
import shutil

def preprocess_home_credit(train_file_path, test_file_path, random_state=42, sample_rows=50000):
    print(f"Starting Home Credit preprocessing (simplified memory-optimized single pass on {sample_rows} rows)...")
    
    # --- Step 1: Infer optimized data types from a sample --- 
    print("Inferring optimized data types from a sample...")
    sample_df = pd.read_csv(train_file_path, nrows=10000)
    inferred_dtype = {}
    for col in sample_df.columns:
        if col == 'SK_ID_CURR':
            inferred_dtype[col] = 'int32'
        elif col == 'TARGET':
            inferred_dtype[col] = 'int8'
        elif sample_df[col].dtype == 'int64':
            inferred_dtype[col] = 'int32'
        elif sample_df[col].dtype == 'float64':
            inferred_dtype[col] = 'float32'
        elif sample_df[col].dtype == 'object':
            inferred_dtype[col] = 'category'
    del sample_df
    gc.collect()

    # --- Step 2: Identify features and high-missingness columns by iterating through chunks ---
    print("Identifying features and high-missingness columns by iterating through chunks...")
    all_columns = pd.read_csv(train_file_path, nrows=0).columns.tolist() 
    missing_counts = pd.Series(0, index=all_columns)
    total_rows = 0

    # Use a smaller chunksize for this step as well
    for chunk in pd.read_csv(train_file_path, chunksize=5000, dtype=inferred_dtype):
        missing_counts += chunk.isnull().sum()
        total_rows += len(chunk)
        del chunk
        gc.collect()

    missing_percentages = (missing_counts / total_rows) * 100
    high_missingness_cols = missing_percentages[missing_percentages > 80].index.tolist()
    
    sample_df_for_features = pd.read_csv(train_file_path, nrows=10000, dtype=inferred_dtype)
    sample_df_for_features = sample_df_for_features.drop(columns=high_missingness_cols + ["SK_ID_CURR", "TARGET"], errors='ignore')

    categorical_features = sample_df_for_features.select_dtypes(include=["category", "object"]).columns.tolist()
    numerical_features = sample_df_for_features.select_dtypes(include=np.number).columns.tolist()
    del sample_df_for_features
    gc.collect()

    final_columns_to_load = ["SK_ID_CURR", "TARGET"] + numerical_features + categorical_features
    final_columns_to_load = list(set(final_columns_to_load)) 

    # --- Step 3: Define preprocessing pipelines and fit on a sample --- 
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Fit a global preprocessor on a sample of the full training data
    print("Fitting global preprocessor on a sample of training data...")
    df_sample_for_preprocessor = pd.read_csv(train_file_path, nrows=sample_rows, dtype=inferred_dtype, usecols=final_columns_to_load)
    X_sample_for_preprocessor = df_sample_for_preprocessor.drop(columns=["TARGET", "SK_ID_CURR"], errors='ignore')
    
    global_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ], remainder="passthrough")
    
    global_preprocessor.fit(X_sample_for_preprocessor)
    del df_sample_for_preprocessor, X_sample_for_preprocessor
    gc.collect()

    ohe_feature_names_global = global_preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + ohe_feature_names_global.tolist()

    # --- Step 4: Process the entire training set in chunks and save to disk ---
    print(f"Processing entire training set in chunks (first {sample_rows} rows)...")
    X_train_processed_list = []
    y_train_list = []
    for chunk in pd.read_csv(train_file_path, nrows=sample_rows, chunksize=5000, dtype=inferred_dtype, usecols=final_columns_to_load):
        X_train_chunk = chunk.drop(columns=["TARGET", "SK_ID_CURR"], errors='ignore')
        y_train_chunk = chunk["TARGET"]
        X_train_processed_chunk = global_preprocessor.transform(X_train_chunk)
        X_train_processed_list.append(pd.DataFrame(X_train_processed_chunk, columns=all_feature_names, index=X_train_chunk.index))
        y_train_list.append(y_train_chunk)
        del chunk, X_train_chunk, y_train_chunk, X_train_processed_chunk
        gc.collect()
    
    X_train_processed = pd.concat(X_train_processed_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    del X_train_processed_list, y_train_list
    gc.collect()

    # --- Step 5: Process the final test set ---
    print(f"Processing final test set (first {sample_rows} rows)...")
    X_test_processed_list = []
    for chunk in pd.read_csv(test_file_path, nrows=sample_rows, chunksize=5000, dtype=inferred_dtype, usecols=[col for col in final_columns_to_load if col != "TARGET"]):
        X_test_raw_chunk = chunk.drop(columns=["SK_ID_CURR"], errors='ignore')
        X_test_processed_chunk = global_preprocessor.transform(X_test_raw_chunk)
        X_test_processed_list.append(pd.DataFrame(X_test_processed_chunk, columns=all_feature_names, index=X_test_raw_chunk.index))
        del chunk, X_test_raw_chunk, X_test_processed_chunk
        gc.collect()

    X_test_processed = pd.concat(X_test_processed_list, ignore_index=True)
    del X_test_processed_list, global_preprocessor
    gc.collect()

    print(f"Preprocessing complete for Home Credit dataset. Processed training data shape: {X_train_processed.shape}, test data shape: {X_test_processed.shape}")
    
    # Return a single 'fold' for compatibility with downstream tasks that expect a list of folds
    # In this simplified version, the 'validation' set is essentially empty or can be derived later if needed.
    folds = [{
        "X_train": X_train_processed,
        "X_val": pd.DataFrame(columns=all_feature_names), # Empty DataFrame for validation set
        "y_train": y_train,
        "y_val": pd.Series(dtype='int8') # Empty Series for validation target
    }]

    return folds, all_feature_names, X_test_processed

if __name__ == "__main__":
    train_path = "/home/ubuntu/explainable-credit-scoring/data/raw/application_train.csv"
    test_path = "/home/ubuntu/explainable-credit-scoring/data/raw/application_test.csv"

    # Process a smaller sample of the Home Credit dataset to avoid memory issues
    folds, feature_names, X_test_processed = preprocess_home_credit(train_path, test_path, sample_rows=50000)

    print(f"Number of features after preprocessing: {len(feature_names)}")
    print(f"Shape of X_train: {folds[0]['X_train'].shape}")
    print(f"Shape of y_train: {folds[0]['y_train'].shape}")
    print(f"Shape of X_val (empty in this simplified version): {folds[0]['X_val'].shape}")
    print(f"Shape of y_val (empty in this simplified version): {folds[0]['y_val'].shape}")
    print(f"Shape of final X_test: {X_test_processed.shape}")

    output_dir = "explainable-credit-scoring/data_preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    processed_train_df_path = os.path.join(output_dir, "home_credit_processed_train_data.csv")
    folds[0]["X_train"].join(folds[0]["y_train"].rename("TARGET")).to_csv(processed_train_df_path, index=False)
    print(f"Processed training data saved to {processed_train_df_path}")

    hc_processed_test_df_path = os.path.join(output_dir, "home_credit_processed_test_data.csv")
    X_test_processed.to_csv(hc_processed_test_df_path, index=False)
    print("Processed test data saved.")

    # Save feature names for later use
    with open(os.path.join(output_dir, "home_credit_feature_names.txt"), "w") as f:
        for item in feature_names:
            f.write(f"{item}\n")
    print("Feature names saved.")

