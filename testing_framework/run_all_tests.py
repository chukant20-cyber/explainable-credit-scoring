import subprocess
import os

def run_all_tests():
    print("--- Running all testing framework scripts ---")

    scripts = [
        "test_data_preprocessing.py",
        "test_model_training.py",
        "test_explanation_generation.py",
        "test_fairness_evaluation.py",
        "test_stability_analysis.py",
    ]

    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        print(f"--- Running {script} ---")
        process = subprocess.run(["python3", script_path], capture_output=True, text=True)
        if process.returncode != 0:
            print(f"Script {script} failed with error code {process.returncode}")
            print(f"Stdout: {process.stdout}")
            print(f"Stderr: {process.stderr}")
        else:
            print(f"Stdout: {process.stdout}")
        print(f"--- {script} finished ---")

    print("--- All testing framework scripts have been attempted ---")

if __name__ == "__main__":
    run_all_tests()

