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
            "Report",
            f"Total records: {total_records}",
            f"Number of missed values in every column:\n{missing_count}",
            f"Total number of missed values: {total_missing_count}"
        ]

        return "\n".join(report)




