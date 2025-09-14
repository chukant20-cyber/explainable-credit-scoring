import os
import pickle

def test_model_training():
    print("Starting model training tests...")

    # Test UCI Credit Card model existence
    uci_model_path = "explainable-credit-scoring/model_training/uci_credit_card_model.pkl"
    if os.path.exists(uci_model_path):
        with open(uci_model_path, 'rb') as f:
            model = pickle.load(f)
        assert model is not None, "UCI Credit Card model not loaded correctly"
        print("UCI Credit Card model training test passed.")
    else:
        print(f"Warning: UCI Credit Card model not found at {uci_model_path}. Skipping test.")

    # Test LendingClub model existence
    lc_model_path = "explainable-credit-scoring/model_training/lending_club_model.pkl"
    if os.path.exists(lc_model_path):
        with open(lc_model_path, 'rb') as f:
            model = pickle.load(f)
        assert model is not None, "LendingClub model not loaded correctly"
        print("LendingClub model training test passed.")
    else:
        print(f"Warning: LendingClub model not found at {lc_model_path}. Skipping test.")

    # Test Home Credit model existence
    hc_model_path = "explainable-credit-scoring/model_training/home_credit_model.pkl"
    if os.path.exists(hc_model_path):
        with open(hc_model_path, 'rb') as f:
            model = pickle.load(f)
        assert model is not None, "Home Credit model not loaded correctly"
        print("Home Credit model training test passed.")
    else:
        print(f"Warning: Home Credit model not found at {hc_model_path}. Skipping test.")

    print("Model training tests complete.")

if __name__ == "__main__":
    test_model_training()

