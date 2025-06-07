# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import os
from pathlib import Path
from data_loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


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
    base_dir = Path.cwd()
    path_to_file = os.path.join(base_dir, "hw4/dataset/creditcard.csv")

    data = load_data(path_to_file)
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
