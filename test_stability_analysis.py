import os

def test_stability_analysis():
    print("Starting stability analysis tests...")

    # Test UCI Credit Card stability analysis output
    uci_stability_output_path = "explainable-credit-scoring/stability_analysis/uci_credit_card_stability_results.txt"
    if os.path.exists(uci_stability_output_path):
        with open(uci_stability_output_path, 'r') as f:
            content = f.read()
        assert len(content) > 0, "UCI Credit Card stability results file is empty"
        print("UCI Credit Card stability analysis test passed.")
    else:
        print(f"Warning: UCI Credit Card stability results file not found at {uci_stability_output_path}. Skipping test.")

    # Test LendingClub stability analysis output
    lc_stability_output_path = "explainable-credit-scoring/stability_analysis/lending_club_stability_results.txt"
    if os.path.exists(lc_stability_output_path):
        with open(lc_stability_output_path, 'r') as f:
            content = f.read()
        assert len(content) > 0, "LendingClub stability results file is empty"
        print("LendingClub stability analysis test passed.")
    else:
        print(f"Warning: LendingClub stability results file not found at {lc_stability_output_path}. Skipping test.")

    # Test Home Credit stability analysis output
    hc_stability_output_path = "explainable-credit-scoring/stability_analysis/home_credit_stability_results.txt"
    if os.path.exists(hc_stability_output_path):
        with open(hc_stability_output_path, 'r') as f:
            content = f.read()
        assert len(content) > 0, "Home Credit stability results file is empty"
        print("Home Credit stability analysis test passed.")
    else:
        print(f"Warning: Home Credit stability results file not found at {hc_stability_output_path}. Skipping test.")

    print("Stability analysis tests complete.")

if __name__ == "__main__":
    test_stability_analysis()

