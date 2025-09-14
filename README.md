# explainable-credit-scoring
Explainable AI for Credit Scoring with SHAP-Calibrated Ensembles.



check the master branch for the main folders
# Explainable Credit Scoring Reproducibility Artifacts

This repository contains reproducibility artifacts for an explainable credit scoring project, covering data preprocessing, model training, explanation generation, fairness evaluation, and statistical testing. The project aims to provide a comprehensive framework for building and evaluating transparent and fair credit scoring models.

## Repository Structure

```
explainable-credit-scoring/
├── data_preprocessing/
│   ├── preprocess_uci_credit_card.py
│   ├── preprocess_lending_club.py
│   ├── preprocess_home_credit.py
│   └── run_all_preprocessing.py
├── model_training/
│   ├── train_uci_credit_card.py
│   ├── train_lending_club.py
│   ├── train_home_credit.py
│   └── run_all_model_training.py
├── explanation_generation/
│   ├── explain_uci_credit_card.py
│   ├── explain_lending_club.py
│   ├── explain_home_credit.py
│   └── run_all_explanation_generation.py
├── fairness_evaluation/
│   ├── evaluate_fairness_uci_credit_card.py
│   ├── evaluate_fairness_lending_club.py
│   ├── evaluate_fairness_home_credit.py
│   └── run_all_fairness_evaluation.py
├── statistical_testing/
│   ├── test_uci_credit_card.py
│   ├── test_lending_club.py
│   ├── test_home_credit.py
│   └── run_all_statistical_testing.py
├── data/
│   └── (raw and processed data files will be stored here)
├── models/
│   └── (trained model files will be stored here)
├── explanations/
│   └── (SHAP values and other explanation outputs will be stored here)
├── fairness_results/
│   └── (fairness evaluation reports will be stored here)
├── statistical_results/
│   └── (statistical test results will be stored here)
└── README.md
└── requirements.txt
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[repository]/explainable-credit-scoring.git
    cd explainable-credit-scoring
    ```
    *(Note: Replace `[repository]` with the actual repository name if available.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data

Due to the size and licensing of some datasets, raw data files are not directly included in this repository. Please download the following datasets and place them in the `data/raw/` directory (create if it doesn't exist):

*   **UCI Credit Card Default Data:** `UCI_Credit_Card.csv`
*   **LendingClub Loan Data:** `loans_full_schema.csv`
*   **Home Credit Default Risk Data:** `application_train.csv`, `application_test.csv`, `HomeCredit_columns_description.csv`

*(Note: During the generation of these artifacts, the Home Credit dataset processing was skipped due to memory constraints in the sandbox environment. The scripts are provided, but full execution may require more resources.)*

## Usage

The project is structured into several modules, each with a `run_all_*.py` script to execute the respective pipeline.

### 1. Data Preprocessing

This module contains scripts for cleaning, transforming, and preparing the raw data for model training. It includes handling missing values, encoding categorical features, and splitting data into training and testing sets using stratified k-fold cross-validation.

To run all data preprocessing steps:
```bash
python3 data_preprocessing/run_all_preprocessing.py
```

Processed data for the first fold of each dataset will be saved in the `data/processed/` directory.

### 2. Model Training

This module implements the training of credit scoring models (e.g., Logistic Regression) on the preprocessed data. It includes hyperparameter specifications and saves the trained models.

To run all model training steps:
```bash
python3 model_training/run_all_model_training.py
```

Trained models will be saved in the `models/` directory.

### 3. Explanation Generation and Stability Analysis

This module focuses on generating explanations for model predictions using techniques like SHAP (SHapley Additive exPlanations). It also includes conceptual frameworks for stability analysis of these explanations.

To run all explanation generation steps:
```bash
python3 explanation_generation/run_all_explanation_generation.py
```

SHAP values and related explanation outputs will be saved in the `explanations/` directory.

### 4. Fairness Evaluation and Optimization Frameworks

This module provides scripts for evaluating the fairness of the trained models with respect to protected attributes (e.g., gender, home ownership) using metrics from libraries like AIF360. It lays the groundwork for potential fairness optimization.

To run all fairness evaluation steps:
```bash
python3 fairness_evaluation/run_all_fairness_evaluation.py
```

Fairness evaluation reports will be saved in the `fairness_results/` directory.

### 5. Statistical Testing Procedures

This module includes scripts for performing statistical tests on model explanations and fairness metrics, with considerations for multiple comparison corrections. This helps in rigorously assessing the significance of findings.

To run all statistical testing steps:
```bash
python3 statistical_testing/run_all_statistical_testing.py
```

Statistical test results will be saved in the `statistical_results/` directory.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
