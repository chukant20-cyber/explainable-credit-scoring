import os

def test_explanation_generation():
    print("Starting explanation generation tests...")

    # Test UCI Credit Card explanation output
    uci_explanation_path = "explainable-credit-scoring/explanation_generation/uci_credit_card_explanations.txt"
    if os.path.exists(uci_explanation_path):
        with open(uci_explanation_path, 'r') as f:
            content = f.read()
        assert len(content) > 0, "UCI Credit Card explanation file is empty"
        print("UCI Credit Card explanation generation test passed.")
    else:
        print(f"Warning: UCI Credit Card explanation file not found at {uci_explanation_path}. Skipping test.")

    # Test LendingClub explanation output
    lc_explanation_path = "explainable-credit-scoring/explanation_generation/lending_club_explanations.txt"
    if os.path.exists(lc_explanation_path):
        with open(lc_explanation_path, 'r') as f:
            content = f.read()
        assert len(content) > 0, "LendingClub explanation file is empty"
        print("LendingClub explanation generation test passed.")
    else:
        print(f"Warning: LendingClub explanation file not found at {lc_explanation_path}. Skipping test.")

    # Test Home Credit explanation output
    hc_explanation_path = "explainable-credit-scoring/explanation_generation/home_credit_explanations.txt"
    if os.path.exists(hc_explanation_path):
        with open(hc_explanation_path, 'r') as f:
            content = f.read()
        assert len(content) > 0, "Home Credit explanation file is empty"
        print("Home Credit explanation generation test passed.")
    else:
        print(f"Warning: Home Credit explanation file not found at {hc_explanation_path}. Skipping test.")

    print("Explanation generation tests complete.")

if __name__ == "__main__":
    test_explanation_generation()

