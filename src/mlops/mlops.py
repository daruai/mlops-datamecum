import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by filling missing values and normalizing.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Fill missing values with the mean of each column
    df.fillna(df.mean(), inplace=True)
    
    # Normalize the data
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df
