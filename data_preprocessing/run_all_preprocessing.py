import os
import pandas as pd
import pickle
from preprocess_uci_credit_card import preprocess_uci_credit_card
from preprocess_lending_club import preprocess_lending_club
from preprocess_home_credit import preprocess_home_credit

def run_all_preprocessing():
    output_dir = "explainable-credit-scoring/data_preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    # --- Preprocessing UCI Credit Card Dataset ---
    print("--- Preprocessing UCI Credit Card Dataset ---")
    uci_credit_card_path = "/home/ubuntu/explainable-credit-scoring/data/raw/UCI_Credit_Card.csv"
    uci_folds, uci_feature_names = preprocess_uci_credit_card(uci_credit_card_path)
    print(f"UCI Credit Card: {len(uci_folds)} folds generated, {len(uci_feature_names) - 1} features.\n")

    # Save preprocessed data for the first fold of UCI Credit Card
    uci_processed_df_path = os.path.join(output_dir, "uci_credit_card_processed_data_fold0.csv")
    processed_train_df_uci = uci_folds[0]["X_train"].copy()
    processed_train_df_uci["TARGET"] = uci_folds[0]["y_train"]
    processed_train_df_uci.to_csv(uci_processed_df_path, index=False)
    print(f"First fold preprocessed data for UCI Credit Card saved to {uci_processed_df_path}")

    uci_folds_info_path = os.path.join(output_dir, "uci_credit_card_folds_info.pkl")
    with open(uci_folds_info_path, "wb") as f:
        pickle.dump([{"train_idx": fold["X_train"].index.tolist(), "test_idx": fold["X_test"].index.tolist()} for fold in uci_folds], f)
    print(f"Folds indices for UCI Credit Card saved to {uci_folds_info_path}\n")

    # --- Preprocessing LendingClub Dataset ---
    print("--- Preprocessing LendingClub Dataset ---")
    lending_club_path = "/home/ubuntu/explainable-credit-scoring/data/raw/loans_full_schema.csv"
    lc_folds, lc_feature_names = preprocess_lending_club(lending_club_path)
    print(f"LendingClub: {len(lc_folds)} folds generated, {len(lc_feature_names)} features.\n")

    # Save preprocessed data for the first fold of LendingClub
    lc_processed_df_path = os.path.join(output_dir, "lending_club_processed_data_fold0.csv")
    processed_train_df_lc = lc_folds[0]["X_train"].copy()
    processed_train_df_lc["TARGET"] = lc_folds[0]["y_train"]
    processed_train_df_lc.to_csv(lc_processed_df_path, index=False)
    print(f"First fold preprocessed data for LendingClub saved to {lc_processed_df_path}")

    lc_folds_info_path = os.path.join(output_dir, "lending_club_folds_info.pkl")
    with open(lc_folds_info_path, "wb") as f:
        pickle.dump([{"train_idx": fold["X_train"].index.tolist(), "test_idx": fold["X_test"].index.tolist()} for fold in lc_folds], f)
    print(f"Folds indices for LendingClub saved to {lc_folds_info_path}\n")

    # --- Preprocessing Home Credit Default Risk Dataset (Skipped) ---
    print("--- Preprocessing Home Credit Default Risk Dataset ---")
    home_credit_train_path = "/home/ubuntu/explainable-credit-scoring/data/raw/application_train.csv"
    home_credit_test_path = "/home/ubuntu/explainable-credit-scoring/data/raw/application_test.csv"
    hc_folds, hc_feature_names, hc_X_test_processed = preprocess_home_credit(home_credit_train_path, home_credit_test_path)
    print(f"Home Credit: {len(hc_folds)} folds generated, {len(hc_feature_names)} features.\n")

    # Save preprocessed data for Home Credit (single pass)
    hc_processed_train_df_path = os.path.join(output_dir, "home_credit_processed_train_data.csv")
    processed_train_df_hc = hc_folds[0]["X_train"].copy()
    processed_train_df_hc["TARGET"] = hc_folds[0]["y_train"]
    processed_train_df_hc.to_csv(hc_processed_train_df_path, index=False)
    print(f"Processed training data for Home Credit saved to {hc_processed_train_df_path}")

    # For Home Credit, since we are doing a single pass, we save a dummy folds info or adapt downstream tasks.
    # For now, we'll save the feature names as an indicator.
    hc_feature_names_path = os.path.join(output_dir, "home_credit_feature_names.txt")
    with open(hc_feature_names_path, "w") as f:
        for item in hc_feature_names:
            f.write(f"{item}\n")
    print(f"Home Credit feature names saved to {hc_feature_names_path}\n")

    # Save processed test data for Home Credit
    hc_processed_test_df_path = os.path.join(output_dir, "home_credit_processed_test_data.csv")
    hc_X_test_processed.to_csv(hc_processed_test_df_path, index=False)
    print(f"Processed test data for Home Credit saved to {hc_processed_test_df_path}\n")

    print("All preprocessing complete.")

if __name__ == "__main__":
    run_all_preprocessing()

