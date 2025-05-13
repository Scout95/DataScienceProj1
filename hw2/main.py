from pathlib import Path
from data_loader import DataLoader
from data_processing import DataProcessing
from missing_values_handler import MissingValuesHandler
import os

def main():
    loader = DataLoader()

    base_dir = Path.cwd()  # or Path(__file__).parent
    # relative_path = os.path.relpath(absolute_path, base_dir)

    # # Load data from csv
    print("base_dir:", base_dir)

    current_dir = os.getcwd()
    print("current_dir:", current_dir)
    path_to_file = os.path.join(base_dir, "hw2/dataset/amazon_sales_data 2025.csv")

    data = loader.load_csv(path_to_file)
    #print(f"Got data: {data}")
    # display 5 row and 11 columns only
    print(data.head())

    df = loader.load_csv_using_row_index(path_to_file, index_col='Order ID')
    print(df.head())

    # prepare the report
    data_processing = DataProcessing()
    # check missing values
    missing_data = data_processing.check_missing_values(df)
    print("missing_data:", missing_data)

    missing_values_handler = MissingValuesHandler(df)
    #missing_values_counted = missing_values_handler.missing_values_report()
    print(missing_values_handler.missing_values_report())



if __name__ == "__main__":
    main()
