import os

def test_fairness_evaluation():
    print("Starting fairness evaluation tests...")

    # Test UCI Credit Card fairness evaluation output
    uci_fairness_output_path = "explainable-credit-scoring/fairness_evaluation/uci_credit_card_fairness_results.txt"
    if os.path.exists(uci_fairness_output_path):
        with open(uci_fairness_output_path, 'r') as f:
            content = f.read()
        assert len(content) > 0, "UCI Credit Card fairness results file is empty"
        print("UCI Credit Card fairness evaluation test passed.")
    else:
        print(f"Warning: UCI Credit Card fairness results file not found at {uci_fairness_output_path}. Skipping test.")

    # Test LendingClub fairness evaluation output
    lc_fairness_output_path = "explainable-credit-scoring/fairness_evaluation/lending_club_fairness_results.txt"
    if os.path.exists(lc_fairness_output_path):
        with open(lc_fairness_output_path, 'r') as f:
            content = f.read()
        assert len(content) > 0, "LendingClub fairness results file is empty"
        print("LendingClub fairness evaluation test passed.")
    else:
        print(f"Warning: LendingClub fairness results file not found at {lc_fairness_output_path}. Skipping test.")

    # Test Home Credit fairness evaluation output
    hc_fairness_output_path = "explainable-credit-scoring/fairness_evaluation/home_credit_fairness_results.txt"
    if os.path.exists(hc_fairness_output_path):
        with open(hc_fairness_output_path, 'r') as f:
            content = f.read()
        assert len(content) > 0, "Home Credit fairness results file is empty"
        print("Home Credit fairness evaluation test passed.")
    else:
        print(f"Warning: Home Credit fairness results file not found at {hc_fairness_output_path}. Skipping test.")

    print("Fairness evaluation tests complete.")

if __name__ == "__main__":
    test_fairness_evaluation()

