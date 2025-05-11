from pathlib import Path
from data_loader import DataLoader
import os

def main():
    # Create an instance of DataLoader
    loader = DataLoader()

    base_dir = Path.cwd()  # or Path(__file__).parent
    # relative_path = os.path.relpath(absolute_path, base_dir)

    # # Load data from csv
    print("base_dir:", base_dir)

    current_dir = os.getcwd()
    print("current_dir:", current_dir)
    path_to_file = os.path.join(base_dir, "hw2/dataset/amazon_sales_data 2025.csv")

    data = loader.load_csv(path_to_file)
    print(f"Got data: {data}")

if __name__ == "__main__":
    main()
