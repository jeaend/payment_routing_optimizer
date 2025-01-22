import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, classification_report, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Preprocesses the dataset for modeling.
    - Drops unnecessary columns.
    - Renames columns to lowercase.
    - Converts specified columns to object type.
    - Adds a new 'is_peak_time' feature.
    
    Parameters:
        df (pd.DataFrame): The input dataset.
        
    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    # Rename all columns to lowercase
    df.columns = df.columns.str.lower()

    # Convert booleans to approprate types
    df['success'] = df['success'].astype(int) # as target
    df['3d_secured'] = df['3d_secured'].astype('object')

    # Only grab the chosen features, reorder so target at end
    df = df[['country', 'card', '3d_secured', 'amount', 'psp', 'success']]
    
    return df

def validate_data(df):
    """
    Validates the DataFrame `df` for the following:
    1. Checks if the columns (except 'amount') are of type 'object'.
    2. Validates the distinct values in each column against predefined valid values.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the data to validate.
        
    Prints:
        - Error messages if any column is not of type 'object' (except 'amount').
        - A list of invalid values for each column if any values are found that do not
          match the predefined valid values.
    """

    # Define the valid values for each column
    valid_values = {
        'country': ['Germany', 'Austria', 'Switzerland'],
        'card': ['Master', 'Diners', 'Visa'],
        '3d_secured': [0, 1],
        'psp': ['UK_Card', 'Simplecard', 'Moneycard', 'Goldcard'],
        'success': [0, 1]
    }

        # Validate the distinct values in each column
    for column, valid_list in valid_values.items():
        unique_values = df[column].unique()
        invalid_values = [value for value in unique_values if value not in valid_list]
        if invalid_values:
            print(f"Invalid values in '{column}': {invalid_values}")
        
    # Check if all columns except 'amount' and 'success' are of type object
    for column in df.columns:
        if column not in ['amount', 'success'] and df[column].dtype != 'object':
            print(f"Error: Column '{column}' is of wrong type.")

    # Check if 'success' and 'amount' are of type int
    if df['success'].dtype != 'int' or df['amount'].dtype != 'int':
        print("Error: 'success' or 'amount' is not of type int.")
    
    print("Validation complete and successful.")

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate model predictions and return key metrics: confusion matrix, precision-recall curve, classification report, 
    best F1-score, and overall precision.
    
    Arguments:
    pipeline : Pipeline object
        The trained model pipeline.
    X_test : DataFrame
        The test features.
    y_test : Series
        The true labels.
    
    Returns:
    None, but prints and displays the evaluation results.
    """
    
    # Make predictions
    y_test_pred = pipeline.predict(X_test)
    y_test_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    # 1. Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # 2. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred_prob)
    pr_auc = auc(recall, precision)

    # Create a figure with 1 row and 2 columns for side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Confusion Matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Failure (0)', 'Success (1)'], 
                yticklabels=['Failure (0)', 'Success (1)'], ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')

    # Plot Precision-Recall Curve
    axes[1].plot(recall, precision, color='blue', lw=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve (PR-AUC = {pr_auc:.4f})')
    axes[1].grid()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

    # 3. Classification Report (Precision, Recall, F1-Score)
    report = classification_report(y_test, y_test_pred, target_names=['Failure (0)', 'Success (1)'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    display(report_df)

    # Find threshold for best F1-Score
    valid_mask = (precision + recall) > 0
    f1_scores = 2 * (precision[valid_mask] * recall[valid_mask]) / (precision[valid_mask] + recall[valid_mask])
    valid_thresholds = thresholds[valid_mask[:-1]]  # Match length with thresholds

    # Find the best threshold
    if f1_scores.size > 0:  # Ensure there's at least one valid F1-score
        best_f1 = f1_scores.max()
        best_threshold = valid_thresholds[f1_scores.argmax()]
        print(f"Best F1-Score: {best_f1:.4f} at Threshold: {best_threshold:.4f}")
    else:
        print("No valid F1-scores could be calculated.")

    # 4. Total F1-Score
    f1 = f1_score(y_test, y_test_pred, pos_label=1)
    print(f"Total Model F1-Score: {f1:.4f}")

    # 5. Overall Precision
    precision = precision_score(y_test, y_test_pred, average='binary', pos_label=1)
    print(f"Total Model Precision: {precision:.4f}")

# Example usage:
# evaluate_model(pipeline, X_test, y_test)

def predict_psp_probabilities(model, preprocessor, new_transactions, psp_list):
    """
    Predict success probabilities for each PSP for a given input data point.
    Returns a interpretable table where it presents feature and prop per PSP
    """
    results = []
    for idx, row in new_transactions.iterrows():
        row_results = []
        for psp in psp_list:
            # Get copy of input, convert to DF to ensure it works with preprocessor
            new_transactions_with_psp = row.copy()
            new_transactions_with_psp['psp'] = psp 
            new_transactions_with_psp_df = pd.DataFrame([new_transactions_with_psp]) 

            # Preprocess the data using the transformer
            preprocessed = preprocessor.transform(new_transactions_with_psp_df) 

            # Predict the probabilities for the 'success' class - index 1
            prob = model.predict_proba(preprocessed)[:, 1][0]  
            
            # Append the result
            row_results.append(prob)
        
        # Append the row results - prob for all PSPs
        results.append(row_results)
    
    return np.array(results)

def log_transform(x):
    return np.log1p(x)