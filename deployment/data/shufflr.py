import pandas as pd
def remove_result_column(df):
    """
    Removes the 'result' column from a DataFrame if it exists.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing weather data
    
    Returns:
        pd.DataFrame: DataFrame with 'result' column removed
    """
    if 'result' in df.columns:
        return df.drop(columns=['result'])
    return df.copy()  # Return copy if no 'result' column exists

# Example usage:
df = pd.read_csv('fakeoutdata.csv')
cleaned_df = remove_result_column(df)
cleaned_df.to_csv('output.csv', index=False)