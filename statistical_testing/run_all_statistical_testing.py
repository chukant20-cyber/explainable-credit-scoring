import os
import subprocess

def run_test_script(script_path):
    print(f"\n--- Running {os.path.basename(script_path)} ---")
    try:
        result = subprocess.run(["python3", script_path], capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Error Output:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Script {os.path.basename(script_path)} failed with error code {e.returncode}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
    except FileNotFoundError:
        print(f"Error: Python interpreter or script not found for {os.path.basename(script_path)}")

if __name__ == "__main__":
    base_dir = "explainable-credit-scoring/statistical_testing"
    
    uci_script = os.path.join(base_dir, "test_uci_credit_card.py")
    lending_club_script = os.path.join(base_dir, "test_lending_club.py")
    home_credit_script = os.path.join(base_dir, "test_home_credit.py")

    run_test_script(uci_script)
    run_test_script(lending_club_script)
    run_test_script(home_credit_script)

    print("\n--- All statistical testing scripts have been attempted ---")

