import os
import pandas as pd
import numpy as np

def test_data_preprocessing():
    print("Starting data preprocessing tests...")

    # Test UCI Credit Card preprocessing output
    uci_processed_path = "explainable-credit-scoring/data_preprocessing/uci_credit_card_processed_data_fold0.csv"
    if os.path.exists(uci_processed_path):
        df_uci = pd.read_csv(uci_processed_path)
        assert not df_uci.isnull().any().any(), "UCI Credit Card processed data contains NaNs"
        assert "TARGET" in df_uci.columns, "UCI Credit Card processed data missing TARGET column"
        print("UCI Credit Card preprocessing test passed.")
    else:
        print(f"Warning: UCI Credit Card processed data not found at {uci_processed_path}. Skipping test.")

    # Test LendingClub preprocessing output
    lc_processed_path = "explainable-credit-scoring/data_preprocessing/lending_club_processed_data_fold0.csv"
    if os.path.exists(lc_processed_path):
        df_lc = pd.read_csv(lc_processed_path)
        assert not df_lc.isnull().any().any(), "LendingClub processed data contains NaNs"
        assert "TARGET" in df_lc.columns, "LendingClub processed data missing TARGET column"
        print("LendingClub preprocessing test passed.")
    else:
        print(f"Warning: LendingClub processed data not found at {lc_processed_path}. Skipping test.")

    # Test Home Credit preprocessing output
    hc_processed_path = "explainable-credit-scoring/data_preprocessing/home_credit_processed_data_fold0.csv"
    if os.path.exists(hc_processed_path):
        df_hc = pd.read_csv(hc_processed_path)
        assert not df_hc.isnull().any().any(), "Home Credit processed data contains NaNs"
        assert "TARGET" in df_hc.columns, "Home Credit processed data missing TARGET column"
        print("Home Credit preprocessing test passed.")
    else:
        print(f"Warning: Home Credit processed data not found at {hc_processed_path}. Skipping test.")

    print("Data preprocessing tests complete.")

if __name__ == "__main__":
    test_data_preprocessing()

