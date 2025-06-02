import os
import subprocess
from pathlib import Path

# import kagglehub
# from kagglehub import KaggleDatasetAdapter
import sqlite3
import pandas as pd

from data_loader import DataLoader
from data_processing import DataProcessing
import visualization
from missing_values_handler import MissingValuesHandler


def download_kaggle_dataset(dataset_name, download_path="dataset"):
    """
    Download a Kaggle dataset using Kaggle API.

    Args:
        dataset_name (str): Kaggle dataset identifier, e.g. 'username/dataset-name'
        download_path (str): Directory to download and extract dataset files
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Kaggle API command to download and unzip dataset
    # --unzip automatically extracts the dataset zip file
    cmd = [
        "kaggle",
        "datasets",
        "download",
        dataset_name,
        "--path",
        download_path,
        "--unzip",
    ]

    print(f"Downloading Kaggle dataset '{dataset_name}' to '{download_path}'...")
    try:
        subprocess.run(cmd, check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")


def create_db_and_table():
    """Create SQLite database and table structure"""
    conn = sqlite3.connect("datascience1.db")
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS top_perfume_brands (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        brand VARCHAR(50),
        title VARCHAR(50),
        type VARCHAR(20),
        price NUMERIC,
        sold INTEGER CHECK(sold >= 0),
        date_of_sale DATE,
        location_of_sale VARCHAR(50)
    );
    """
    )
    conn.commit()
    conn.close()


def load_and_clean_data(file_path):
    """Load and preprocess dataset"""
    # df = pd.read_csv(file_path)

    loader = DataLoader()
    base_dir = Path.cwd()  # or Path(__file__).parent

    # # Load data from csv
    current_dir = os.getcwd()
    path_to_file = os.path.join(base_dir, "dataset/ebay_womens_perfume.csv")

    try:
        df = loader.load_csv(path_to_file)
        print(df.head())
    except RuntimeError as e:
        print(e)
        return

    # Prepare the report about missing values
    data_processing = DataProcessing()
    # check missing values
    missing_data = data_processing.check_missing_values(df)
    print("-------------------> Found missing_data: ------------------>")
    print(missing_data)

    missing_values_handler = MissingValuesHandler(df)
    print(missing_values_handler.missing_values_report())

    # fill missed data by mean value
    filled_by_mean = missing_values_handler.fill_missing_values(df, "mean")
    print("-------------------> Filled missed data by mean value: ------------------>")
    print(filled_by_mean)

    # Handle missing values for numeric columns
    for col in ["price", "sold"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
        else:
            print(f"Warning: Column '{col}' not found in dataset.")

    # Remove timezone abbreviation (e.g. 'PDT')
    if "lastUpdated" in df.columns:
        df["lastUpdated_clean"] = df["lastUpdated"].str.replace(
            r"\s[A-Z]{3}$", "", regex=True
        )
        df["date_of_sale"] = pd.to_datetime(df["lastUpdated_clean"], errors="coerce")
    else:
        print("Warning: 'lastUpdated' column not found in dataset.")
        df["date_of_sale"] = pd.NaT  # assign NaT if missing

    return df


def insert_data_to_db(df):
    """Insert cleaned data into database"""
    conn = sqlite3.connect("datascience1.db")
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute(
            """
        INSERT INTO top_perfume_brands 
        (brand, title, type, price, sold, date_of_sale, location_of_sale)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                row.get("brand"),
                row.get("title"),
                row.get("type"),
                row.get("price"),
                int(row.get("sold", 0)),
                (
                    row["date_of_sale"].strftime("%Y-%m-%d")
                    if pd.notnull(row["date_of_sale"])
                    else None
                ),
                row.get("itemLocation"),
            ),
        )

    conn.commit()
    conn.close()


QUERIES = {
    "top_10_brands": "SELECT * FROM top_perfume_brands LIMIT 10;",
    "top_10_sold": """
    SELECT * FROM top_perfume_brands 
    ORDER BY sold DESC LIMIT 10;
    """,
    "top_10_price_asc": """
    SELECT * FROM top_perfume_brands 
    ORDER BY price DESC LIMIT 10;
    """,
    "top_10_sold_grouped_location": """
    SELECT location_of_sale, brand, title, type, price, sold, date_of_sale
    FROM top_perfume_brands
    WHERE sold IN (
        SELECT sold FROM top_perfume_brands 
        ORDER BY sold DESC LIMIT 10
    )
    GROUP BY location_of_sale, brand, title, type, price, sold, date_of_sale
    ORDER BY price ASC
    LIMIT 10;
    """,
}


def execute_queries():
    """Execute all SQL queries and return results"""
    conn = sqlite3.connect("datascience1.db")
    cursor = conn.cursor()
    results = {}

    for name, query in QUERIES.items():
        cursor.execute(query)
        results[name] = {
            "columns": [desc[0] for desc in cursor.description],
            "data": cursor.fetchall(),
        }

    conn.close()
    return results

    # Aggregate SQL queries for data science analysis


AGGREGATE_QUERIES = {
    "Average Price by Brand": """
        SELECT brand, ROUND(AVG(price)) AS avg_price
        FROM top_perfume_brands
        GROUP BY brand
        ORDER BY avg_price DESC
        LIMIT 10;
    """,
    "Total Units Sold by Brand": """
        SELECT brand, SUM(sold) AS total_units_sold
        FROM top_perfume_brands
        GROUP BY brand
        ORDER BY total_units_sold DESC
        LIMIT 10;
    """,
    "Count of Sales Records by Location": """
        SELECT location_of_sale, COUNT(*) AS sales_count
        FROM top_perfume_brands
        GROUP BY location_of_sale
        ORDER BY sales_count DESC
        LIMIT 10;
    """,
    "Average Price and Total Sold by Perfume Type": """
        SELECT type, ROUND(AVG(price)) AS avg_price, SUM(sold) AS total_sold
        FROM top_perfume_brands
        GROUP BY type
        ORDER BY total_sold DESC;
    """,
    "Total Revenue by Brand": """
        SELECT brand, ROUND(SUM(price * sold)) AS total_revenue
        FROM top_perfume_brands
        GROUP BY brand
        ORDER BY total_revenue DESC
        LIMIT 10;
    """,
    "Count of Unique Perfume Titles per Brand": """
        SELECT brand, COUNT(DISTINCT title) AS unique_perfumes
        FROM top_perfume_brands
        GROUP BY brand
        ORDER BY unique_perfumes DESC
        LIMIT 10;
    """,
    "Average Sold Units per Sale by Location": """
        SELECT location_of_sale, ROUND(AVG(sold)) AS avg_units_sold
        FROM top_perfume_brands
        GROUP BY location_of_sale
        ORDER BY avg_units_sold DESC
        LIMIT 10;
    """,
    # "Monthly Total Sales": """
    #     SELECT strftime('%Y-%m', date_of_sale) AS sale_month, SUM(sold) AS total_sold
    #     FROM top_perfume_brands
    #     GROUP BY sale_month
    #     ORDER BY sale_month;
    # """,
}


def run_aggregate_queries():
    """Execute aggregate queries and print results"""
    conn = sqlite3.connect("datascience1.db")
    cursor = conn.cursor()

    for description, query in AGGREGATE_QUERIES.items():
        print(f"\n=== {description} ===")
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        # Print column headers
        print("\t".join(columns))
        # Print rows
        for row in rows:
            print("\t".join(str(item) if item is not None else "NULL" for item in row))

    conn.close()


# if __name__ == "__main__":
#     create_db_and_table()

#     # Load dataset (from actual Kaggle file path) and clean data
#     # df = load_and_clean_data('perfume_ecommerce.csv')
#     df = load_and_clean_data("dataset/ebay_womens_perfume.csv")

#     # Insert data
#     insert_data_to_db(df)

#     # Execute and display queries
#     results = execute_queries()
#     for query_name, result in results.items():
#         print(f"\n{query_name.upper()} RESULTS:")
#         print(f"Columns: {result['columns']}")
#         for row in result["data"]:
#             print(row)

#     run_aggregate_queries()

#     visualization.run_all_plots('datascience1.db')

if __name__ == "__main__":
    # # Example Kaggle dataset identifier (replace with your actual dataset)
    kaggle_dataset = "kanchana1990/perfume-e-commerce-dataset-2024"

    # # Download dataset if not already downloaded
    if not os.path.exists("dataset/ebay_womens_perfume.csv"):
        download_kaggle_dataset(kaggle_dataset, download_path="dataset")

    # # Kaggle dataset slug and file path
    # dataset_slug = "kanchana1990/perfume-e-commerce-dataset-2024"
    # file_path = "ebay_womens_perfume.csv"

    # print("Loading dataset from Kaggle using KaggleHub...")
    # df = kagglehub.load_dataset(
    #     KaggleDatasetAdapter.PANDAS,
    #     dataset_slug,
    #     file_path
    # )
    # print(f"Dataset loaded with {len(df)} records.")

    # Proceed with your existing workflow
    create_db_and_table()
    df = load_and_clean_data("dataset/ebay_womens_perfume.csv")
    insert_data_to_db(df)
    execute_queries()
    run_aggregate_queries()

    # Visualization call if any
    import visualization

    visualization.run_all_plots("datascience1.db")
