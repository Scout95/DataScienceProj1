import pandas as pd

class DataProcessing:
    @staticmethod
    # method takes a single argument, df (expected to be a DataFrame)
    # returns the count of missing values (i.e. null) in each column of the DataFrame
    def check_missing_values(df):
        return df.isnull().sum()

    @staticmethod
    # method takes a single argument, df (expected to be a DataFrame)
    # removes any rows from the DataFrame that contain missing (null or NaN) values
    def drop_missing_values(df):
        return df.dropna()

    @staticmethod
    # returns a new DataFrame where all missing df values (NaN or None) are replaced by the given value
    def fill_missing_values(df, value):
        return df.fillna(value)

    @staticmethod
    # returns a new DataFrame with duplicate rows removed
    def remove_duplicates(df):
        return df.drop_duplicates()

    @staticmethod
    # column: the name (string) of the column in df whose type need to convert
    # dtype: the target data type to convert the column to (int, float, str etc.)
    # returns the the column converted to the new type
    def change_column_type(df, column, dtype):
        try:
            df[column] = df[column].astype(dtype)
            return df
        except Exception as e:
            return f"Error while transforming the type: {e}"