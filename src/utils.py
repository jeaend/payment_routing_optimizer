import pandas as pd
from scipy import stats
import numpy as np

def cramers_v(chi2_stat, n, k, r):
    """Calculate Cramér's V statistic."""
    return np.sqrt(chi2_stat / (n * min(k - 1, r - 1)))

def chi_square_test_multiple(df, alpha=0.05):
    """
    Perform Chi-Square tests on all pairs of object-type features in a DataFrame and identify connected features.
    Make sure that all features that should be considered are of type object.
    Also returns the statistically associated features.

    Parameters:
        df (DataFrame): The input DataFrame.
        alpha (float): The significance level for the test (default is 0.05).

    Returns:
        results_df (DataFrame): A DataFrame containing Chi-Square statistics, p-values, and hypothesis test results.
        associated_features (set): A set of tuples containing pairs of connected features.
        relevant_contingency_tables (dict): A dictionary where the keys are tuples of feature names that have significant associations, and the values are their corresponding contingency tables.
    """

    # Store results
    results = []
    associated_features = set()
    relevant_contingency_tables = {}
    processed_pairs = set() 

    # Explanation about the meaning of the test and the Hypotheses
    print(
        "Null Hypothesis: There is no significant association between the two variables being tested."
    )
    print(
        "Alternative Hypothesis: There is a significant association between the two variables."
    )

    objects = df.select_dtypes(include=["object"]).columns

    # Iterate over all object-type columns
    for col1 in objects:
        for col2 in objects:
            # Do not repeat yourself
            if col1 != col2:
                pair = tuple(sorted([col1, col2]))  # Sorting ensures no duplicates like (A, B) and (B, A)
                if pair in processed_pairs:
                    continue  # Skip if the pair is already processed

                # Create contingency table
                contingency_table = pd.crosstab(df[col1], df[col2])

                # Perform the Chi-Square test
                chi2_stat, p_val, dof, expected = stats.chi2_contingency(
                    contingency_table
                )

                # Determine hypothesis result
                if p_val < alpha:
                    result = "Reject the null hypothesis (Significant association)"
                    associated_features.add(pair)                    
                    relevant_contingency_tables[pair] = contingency_table
                    # Calculate Cramér's V for significant results
                    n = contingency_table.sum().sum()  # Total number of observations
                    k = contingency_table.shape[0]  # Number of categories in the first variable
                    r = contingency_table.shape[1]  # Number of categories in the second variable
                    cramers_v_value = cramers_v(chi2_stat, n, k, r)
                else:
                    result = "Fail to reject the null hypothesis (No significant association)"
                    is_significant = "No"
                    cramers_v_value = None 

                # Append results to the list
                results.append(
                    {
                        "Feature 1": col1,
                        "Feature 2": col2,
                        "Chi2 Stat": chi2_stat,
                        "p-value": p_val,
                        "Degrees of Freedom": dof,
                        "Test Result": result,
                        "Is Significant": is_significant,  
                        "Cramér's V": cramers_v_value,  
                    }
                )
                processed_pairs.add(pair)
    # Convert results into DF
    results_df = pd.DataFrame(results)

    return results_df, associated_features, relevant_contingency_tables


def calculate_transaction_metrics(grouped):
    """
    Calculate transaction metrics for successful and unsuccessful transactions.
    
    Parameters:
    - grouped: DataFrame grouped by 'order_id' and a column (e.g., 'card', 'psp', 'country') with the columns: 
               'amount', 'fee', 'success'
    
    Returns:
    - A dictionary containing the calculated metrics.
    """
    # Amount Sum
    successful_amount_sum = grouped[grouped["success"] == True]["amount"].sum()
    successful_fees_sum = grouped[grouped["success"] == True]["fee"].sum()

    unsuccessful_amount_sum = grouped[grouped["success"] == False]["amount"].sum()
    unsuccessful_fees_sum = grouped[grouped["success"] == False]["fee"].sum()

    # Average Amount
    successful_amount_avg = grouped[grouped["success"] == True]["amount"].mean()
    unsuccessful_amount_avg = grouped[grouped["success"] == False]["amount"].mean()

    # Transaction Count
    successful_transaction_count = grouped[grouped["success"] == True].shape[0]
    unsuccessful_transaction_count = grouped[grouped["success"] == False].shape[0]

    # Average Fees per Transaction
    successful_fees_avg = grouped[grouped["success"] == True]["fee"].mean()
    unsuccessful_fees_avg = grouped[grouped["success"] == False]["fee"].mean()

    # Fees to Amount Ratio (%)
    successful_fee_to_amount_ratio = (successful_fees_sum / successful_amount_sum * 100) if successful_amount_sum != 0 else 0
    unsuccessful_fee_to_amount_ratio = (unsuccessful_fees_sum / unsuccessful_amount_sum * 100) if unsuccessful_amount_sum != 0 else 0

    # Total transaction count
    total_transactions = successful_transaction_count + unsuccessful_transaction_count

    # Percentage of successful transactions
    success_percentage = (successful_transaction_count / total_transactions * 100) if total_transactions != 0 else 0
    unsuccessful_percentage = 100 - success_percentage
    
    # Return the results as a dictionary
    return {
        'Amount Sum (Successful)': successful_amount_sum,
        'Fees Sum (Successful)': successful_fees_sum,
        'Amount Sum (Unsuccessful)': unsuccessful_amount_sum,
        'Fees Sum (Unsuccessful)': unsuccessful_fees_sum,
        'Average Amount (Successful)': successful_amount_avg,
        'Average Amount (Unsuccessful)': unsuccessful_amount_avg,
        'Transaction Count (Successful)': successful_transaction_count,
        'Transaction Count (Unsuccessful)': unsuccessful_transaction_count,
        'Average Fees per Transaction (Successful)': successful_fees_avg,
        'Average Fees per Transaction (Unsuccessful)': unsuccessful_fees_avg,
        'Fees to Amount Ratio (%) (Successful)': successful_fee_to_amount_ratio,
        'Fees to Amount Ratio (%) (Unsuccessful)': unsuccessful_fee_to_amount_ratio,
        'Success Percentage % (Successful)': success_percentage,
        'Success Percentage % (Unsuccessful)': unsuccessful_percentage
    }

def calculate_grouped_metrics(df, group_by_column):
    """
    Calculate transaction metrics grouped by a specified column (card, psp, or country).
    
    Parameters:
    - df: DataFrame containing transaction data with columns: 'order_id', 'amount', 'fee', 'psp', 'card', 'country', 'success'
    - group_by_column: The column by which to group the data (should be one of 'card', 'psp', or 'country')
    
    Returns:
    - A DataFrame containing the calculated metrics for each group.
    """
    # Check if the provided column exists in the DataFrame
    if group_by_column not in df.columns:
        raise ValueError(f"Column '{group_by_column}' does not exist in the DataFrame.")
    
    # Group by the specified column and 'order_id', summing 'amount' and 'fees'
    grouped = df.groupby(["order_id", group_by_column]).agg({
        "amount": "first",  
        "fee": "sum",      
        "success": "first"
    }).reset_index()

    # Calculate the total amount and total transaction count across all groups
    total_amount = grouped["amount"].sum()
    total_transactions = grouped.shape[0]

    # Create an empty list to store the results for each group
    results = []

    # Iterate over each unique value in the specified column
    for group_value in grouped[group_by_column].drop_duplicates().values:
        group_data = grouped[grouped[group_by_column] == group_value]
        
        # Call the calculate_transaction_metrics function
        metrics = calculate_transaction_metrics(group_data)
        group_amount_sum = group_data["amount"].sum()
        amount_percentage = (group_amount_sum / total_amount * 100) if total_amount != 0 else 0
        group_transaction_count = group_data.shape[0]
        transaction_percentage = (group_transaction_count / total_transactions * 100) if total_transactions != 0 else 0
        
        # Add group value, amount percentage, and transaction percentage to the metrics
        metrics[group_by_column] = group_value
        metrics["Amount Percentage of Total (%)"] = amount_percentage
        metrics["Transaction Count Percentage of Total (%)"] = transaction_percentage
        
        # Append the metrics to the results list
        results.append(metrics)
    
    # Convert the list of results into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to make group_by_column the first column
    columns_order = [group_by_column] + [col for col in results_df.columns if col != group_by_column]
    results_df = results_df[columns_order]
    
    # Format numbers for display
    pd.set_option('display.float_format', '{:,.1f}'.format)

    return results_df

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

    # Add the 'is_peak_time' feature (1 if time is between 13:00 and 18:00, else 0)
    df['is_peak_time'] = df['tmsp'].apply(lambda x: 1 if 13 <= x.hour < 18 else 0)
    
    # Convert booleans to approprate types
    df['success'] = df['success'].astype(int) # as target
    df['3d_secured'] = df['3d_secured'].astype('object')
    df['is_peak_time'] = df['is_peak_time'].astype('object')

    # Only grab the chosen features, reorder so target at end
    df = df[['country', 'card', '3d_secured', 'is_peak_time', 'amount', 'psp', 'success']]
    
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
        'is_peak_time': [0, 1],
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
