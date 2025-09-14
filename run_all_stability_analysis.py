import subprocess
import os

def run_stability_analysis():
    print("--- Running all stability analysis scripts ---")

    scripts = [
        "stability_analysis_uci_credit_card.py",
        "stability_analysis_lending_club.py",
        "stability_analysis_home_credit.py",
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

    print("--- All stability analysis scripts have been attempted ---")

if __name__ == "__main__":
    run_stability_analysis()

