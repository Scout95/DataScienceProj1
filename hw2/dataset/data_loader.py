import pandas as pd
import requests

class DataLoader:
   @staticmethod
   def load_csv(file_path):
       try:
           return pd.read_csv(file_path)
       except Exception as e:
           return f"Error while loaading the csv file!: {e}"
   @staticmethod
   def load_json(file_path):
       try:
           return pd.read_json(file_path)
       except Exception as e:
           return f"Error while loaading the JSON: {e}"
   @staticmethod
   def load_api(url, params=None):
       try:
           response = requests.get(url, params=params)
           response.raise_for_status()
           return response.json()
       except Exception as e:
           return f"Error while loaading the data from API: {e}"