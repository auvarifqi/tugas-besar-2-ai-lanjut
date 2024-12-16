import pandas as pd

def duplicate_stats(df):
    # Find the duplicated rows
    duplicated_rows = df[df.duplicated()]

    # Find the duplicated columns
    duplicated_columns = df.columns[df.columns.duplicated()]

    return duplicated_rows, duplicated_columns


def check_negative_values(df, numerical_columns):
    negative_values = {}
    for col in numerical_columns:
        negative_values[col] = df[df[col] < 0].shape[0]

    return negative_values


def missing_value_stats(df):
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Create a DataFrame with missing counts and percentages
    missing_data = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': missing_percentage.round(2).astype(str) + '%',
    })

    return missing_data