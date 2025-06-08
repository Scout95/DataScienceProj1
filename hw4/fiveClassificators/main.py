import pandas as pd
import numpy as np
import os
import zipfile
from pathlib import Path
from data_loader import DataLoader
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def download_dataset():
    dataset_dir = Path("../dataset")
    dataset_dir.mkdir(exist_ok=True)
    zip_path = dataset_dir / "creditcard.zip"

    if not zip_path.exists():
        print("[INFO] Downloading dataset...")
        exit_code = os.system(f'kaggle datasets download -d mlg-ulb/creditcardfraud -p "{dataset_dir}" --force')
        if exit_code != 0:
            raise RuntimeError("Kaggle dataset download failed.")
    else:
        print("[INFO] Dataset zip already exists.")

    return zip_path, dataset_dir

def extract_dataset(zip_path, extract_to):
    if not (extract_to / "creditcard.csv").exists():
        print(f"[INFO] Extracting dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print("[INFO] Extraction complete.")
    else:
        print("[INFO] Dataset already extracted.")


# def download_and_extract_dataset():
#     # Download dataset using kaggle CLI
#     print("[INFO] Downloading dataset from Kaggle...")
#     os.system(
#         "kaggle datasets download -d mlg-ulb/creditcardfraud -p ./dataset --force"
#     )

#     zip_path = Path("./dataset/creditcardfraud.zip")
#     extract_path = Path("./dataset")

#     print("[INFO] Extracting dataset...")
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extract_path)
#     print("[INFO] Dataset extracted.")


def load_data(path: str) -> pd.DataFrame:
    loader = DataLoader()
    try:
        print(f"[INFO] Loading data from file: {path}")
        data = loader.load_csv(path)
        print(f"[INFO] Data successfully loaded. Size: {data.shape}")
        print("Class distribution:\n", data["Class"].value_counts())
        print("First 5 rows:\n", data.head())
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None
    return data


def main():
    try:
        base_dir = Path(__file__).parent.resolve()  # If running as script
    except NameError:
        base_dir = Path(os.getcwd()).resolve()      # If running in notebook

    dataset_dir = base_dir / "../dataset"
    # try:
    #     dataset_dir.mkdir(exist_ok=True)
    #     print(f"'dataset' folder created at: {dataset_dir}")
    # except Exception as e:
    #     print(f"Failed to create dataset directory: {e}")

    zip_path = dataset_dir / "creditcardfraud.zip"
    csv_file = dataset_dir / "creditcard.csv"
    print(f"'zip_path' folder expected at: {zip_path}")
    print(f"'csv_file' folder expected at: {csv_file}")

    # Download dataset if ZIP not present
    if not zip_path.exists():
        print(f"[ERROR] Dataset zip file not found at {zip_path}. Please download it first.")
        # download_and_extract_dataset()
        # download_dataset() #uncomment to download
    # else:
    #     print(f"[INFO] Extracting dataset from {zip_path}...")
    #     with zipfile.ZipFile(zip_path, "r") as zip_ref:
    #         zip_ref.extractall(dataset_dir)
    #     print("[INFO] Dataset extracted.")

    # Extract dataset if CSV not present
    extract_dataset(zip_path, dataset_dir)

    # base_dir = Path.cwd()
    # path_to_file = os.path.join(base_dir, "hw4/dataset/creditcard.csv")
    # data = load_data(path_to_file)

    # Download latest version
    # path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    # print("Path to dataset files:", path)
    # url = "https://raw.githubusercontent.com/jonwancodes/Credit-Card-Fraud-Detection-Dataset-2023-Analysis/main/creditcard.csv"


    # ---
    # dataset_dir = Path("./dataset")
    # csv_file = dataset_dir / "creditcard.csv"

    # Download and extract if dataset not exists
    # if not csv_file.exists():
    #     download_and_extract_dataset()

    data = load_data(csv_file)

    if data is None:
        print("[ERROR] Data loading failed. Exiting.")
        return

    # Pre-analysing
    print("[INFO] Preview of data:")
    print(data.head())
    print("[INFO] Class distribution:")
    print(data["Class"].value_counts())

    # Split data for classes and goal variable
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # Split to train and test selection with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled numpy arrays back to DataFrame with original column names
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=X.columns, index=X_train.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=X.columns, index=X_test.index
    )

    # initialization of classes
    classifiers = {
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "ExtraTrees": ExtraTreesClassifier(random_state=42),
        "QDA": QuadraticDiscriminantAnalysis(),
        "LightGBM": LGBMClassifier(random_state=42),
    }

    # Train and prediction
    results = {}
    for name, clf in classifiers.items():
        print(f"[INFO] Training {name}...")
        # # Use DataFrame for LightGBM, NumPy arrays or DataFrames for others (all accept DataFrame)
        # if name == "LightGBM":
        #     clf.fit(X_train_scaled_df, y_train)
        #     y_pred = clf.predict(X_test_scaled_df)
        #     y_proba = clf.predict_proba(X_test_scaled_df)[:, 1]
        # else:
        #     clf.fit(X_train_scaled_df, y_train)
        #     y_pred = clf.predict(X_test_scaled_df)
        #     if hasattr(clf, "predict_proba"):
        #         y_proba = clf.predict_proba(X_test_scaled_df)[:, 1]
        #     else:
        #         y_proba = None

        clf.fit(X_train_scaled_df, y_train)

        # use DataFrame
        y_pred = clf.predict(X_test_scaled_df)

        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test_scaled_df)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            y_proba = None
            roc_auc = None

        # roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {"roc_auc": roc_auc, "report": report}

    # Results
    for name, metrics in results.items():
        print(f"\n=== Results for {name} ===")
        if metrics["roc_auc"] is not None:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        else:
            print("ROC AUC: Not available")
        print("Classification report:")
        print(
            classification_report(y_test, classifiers[name].predict(X_test_scaled_df))
        )


if __name__ == "__main__":
    main()
