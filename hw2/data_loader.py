import pandas as pd
import requests

class DataLoader:
    @staticmethod
    def load_csv(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            return f"Error while loading the csv file!: {e}"

    @staticmethod
    def load_csv_using_row_index(file_path, index_col):
        try:
            # index_col pass a column name (string) or index (int), not a set!
            return pd.read_csv(file_path, index_col=index_col)
        except Exception as e:
            return f"Error while loading the csv file!: {e}"

    @staticmethod
    def load_json(file_path):
        try:
            return pd.read_json(file_path)
        except Exception as e:
            return f"Error while loading the JSON: {e}"

    @staticmethod
    def load_api(url, params=None):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return f"Error while loading the data from API: {e}"
