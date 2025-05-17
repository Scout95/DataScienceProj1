import pandas as pd


class MissingValuesHandler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # return pandas Series with missing numbers by column
    def count_missing_values(self) -> pd.Series:
        missing_values_counted = self.df.isnull().sum()
        return missing_values_counted

    def missing_values_report(self) -> str:
        total_records = self.df.size
        missing_count = self.count_missing_values()
        total_missing_count = missing_count.sum()

        report = [
            "---------------------->> Report (missing_values_report) >>-----------------------",
            f"Total records: {total_records}",
            f"Number of missed values in every column:\n{missing_count}",
            f"Total number of missed values: {total_missing_count}",
        ]

        return "\n".join(report)

    def fill_missing_values(self, df, type):
        """
        Reads a CSV file and fills missing values in every column using:
        1. Mean (for numeric columns)
        2. Median (for numeric columns)
        3. Mode (for all columns)
        Returns one of three DataFrames: mean_filled, median_filled, mode_filled.
        """
        # Fill with mean (numeric columns only)
        filled_df = df.copy()

        if type == "mean":
            # Fill numeric columns with mean
            for col in filled_df.select_dtypes(include="number").columns:
                mean_value = filled_df[col].mean()
                #filled_df[col].fillna(mean_value, inplace=True)
                filled_df[col].fillna(mean_value)

        elif type == "median":
            # Fill numeric columns with median
            for col in filled_df.select_dtypes(include="number").columns:
                median_value = filled_df[col].median()
                #filled_df[col].fillna(median_value, inplace=True)
                filled_df[col].fillna(median_value)
        elif type == "mode":
            # Fill all columns with mode
            for col in filled_df.columns:
                mode_value = filled_df[col].mode()
                if not mode_value.empty:
                    #filled_df[col].fillna(mode_value[0], inplace=True)
                    filled_df[col].fillna(mode_value)
        else:
            raise ValueError("Parameter 'type' must be one of: 'mean', 'median', 'mode'")
        return filled_df
    
