import os
import csv
import json
import pandas as pd 

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

eval_file_path = "../models/evaluation/model_evaluation_log.csv"

# Initialize the CSV file with headers if it doesn't exist
# If it exists, it appends the data
def initialize_log(file_path=eval_file_path):
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, mode='x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Model_Name", "Confusion_Matrix", "PR_AUC", "F1_Score_Overall", 
                "Precision_Overall", "Recall_Overall", "Class_Metrics", 
                "Train_Score", "Validation_Score", "Notes", "Hyperparameters"  # Hyperparameters at the end
            ])
    except FileExistsError:
        print(f"{file_path} already exists. New entries will be appended.")

# Function to extract metrics
def extract_metrics(pipeline, X_test, y_test):
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Precision-Recall Curve and PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall_curve, precision_curve)

    # Overall Metrics
    f1_overall = f1_score(y_test, y_pred, pos_label=1)
    precision_overall = precision_score(y_test, y_pred, pos_label=1)
    recall_overall = recall_score(y_test, y_pred, pos_label=1)

    # Class-Specific Metrics
    class_report = classification_report(y_test, y_pred, target_names=['Failure (0)', 'Success (1)'], output_dict=True)

    return conf_matrix, pr_auc, f1_overall, precision_overall, recall_overall, class_report

# Function to extract hyperparameters dynamically
def extract_hyperparameters(model):
    """
    Extracts hyperparameters in a JSON-serializable format.
    """
    try:
        params = model.get_params()
        return {k: str(v) for k, v in params.items()}  # Convert all values to strings
    except AttributeError:
        return {"error": "Hyperparameters could not be extracted"}


# Log the results
def log_results(model_name, conf_matrix, pr_auc, f1_overall, 
                precision_overall, recall_overall, class_metrics, 
                train_score, validation_score, notes, hyperparameters, 
                file_path=eval_file_path):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            model_name,
            json.dumps(conf_matrix.tolist()), 
            pr_auc,
            f1_overall,
            precision_overall,
            recall_overall,
            json.dumps(class_metrics),  
            train_score,
            validation_score,
            notes,
            json.dumps(hyperparameters) 
        ])

def log_model_evaluation(model, X_train, X_test, y_train, y_test, model_name, notes, file_path=eval_file_path):
    """
    This function performs the evaluation of the model, extracts metrics, and logs the results.

    Arguments:
    - model : Trained model
    - X_train : Training features
    - X_test : Test features
    - y_train : Training labels
    - y_test : Test labels
    - model_name : Name of the model
    - notes : Additional notes for the model
    - file_path : Path to the CSV log file
    """
    # Extract metrics
    conf_matrix, pr_auc, f1_overall, precision_overall, recall_overall, class_metrics = extract_metrics(model, X_test, y_test)

    # Extract hyperparameters dynamically
    hyperparameters = extract_hyperparameters(model)

    # Initialize log
    initialize_log(file_path)  # Call function to create log file if it doesn't exist

    # Log the results
    log_results(
        model_name=model_name,
        conf_matrix=conf_matrix,
        pr_auc=pr_auc,
        f1_overall=f1_overall,
        precision_overall=precision_overall,
        recall_overall=recall_overall,
        class_metrics=class_metrics,
        train_score=model.score(X_train, y_train),
        validation_score=model.score(X_test, y_test),
        notes=notes,
        hyperparameters=hyperparameters,
        file_path=file_path
    )

# Pretty display of logged results
def display_log(file_path=eval_file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        pd.set_option("display.max_columns", None)  
        print("\n=== Model Evaluation Log ===\n")
        print(df.to_markdown(index=False)) 
    else:
        print(f"No log file found at {file_path}")

# Example Usage
if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    # Generate synthetic data
    X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    pipeline = RandomForestClassifier(n_estimators=100, max_depth=10)
    pipeline.fit(X_train, y_train)

    # Log the evaluation of the model
    log_model_evaluation(
        model=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_name="xgboost_baseline_model",
        notes="Baseline Model, Vanilla XGBoost."
    )
    # Display markdown 
    # display_log()
    