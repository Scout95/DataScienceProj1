import pandas as pd
import numpy as np
import os
from pathlib import Path
from data_loader import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from bayes_opt import BayesianOptimization
import joblib  # for saving model


def load_data(path: str) -> pd.DataFrame:
    loader = DataLoader()
    base_dir = Path.cwd()  # or Path(__file__).parent

    # # Load data from csv
    current_dir = os.getcwd()
    # path_to_file = os.path.join(base_dir, "hw4/dataset/creditcard_2023.csv")
    path_to_file = os.path.join(base_dir, "./dataset/creditcard.csv")

    try:
        print(f"[INFO] Loading data from file: {path_to_file}")
        df = loader.load_csv(path_to_file)
        print(f"[INFO] Data successfully loaded. Size: {df.shape}")
        print(df["Class"].value_counts())
        print("There are heading and first 5 rows of table:")
        print(df.head())
        print("[INFO] Distribution of classes:\n", df["Class"].value_counts())
    except RuntimeError as e:
        print(e)
        return

    # df = pd.read_csv(path)
    return df


def preprocess_and_split(df: pd.DataFrame):
    """
    Data preprocessing: deviding to X and y, standartization Amount, classes balancing
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    print(f"[INFO] Size of selection before balancing: {X.shape}")
    print(f"[INFO] Class distribution before balancing:\n{y.value_counts()}")

    # Split into train/test before balancing and scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"[INFO] Size of train: {X_train.shape}, Size of test: {X_test.shape}")
    print(f"[INFO] Classes in train:\n{y_train.value_counts()}")
    print(f"[INFO] Classes in test:\n{y_test.value_counts()}")

    # 2. Scaling train only
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test_scaled["Amount"] = scaler.transform(X_test[["Amount"]])

    # 3. Undersample train only (DO NOT OVERSAMPLE!)
    print("[INFO] Undersampling train only...")
    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
    X_train_bal, y_train_bal = rus.fit_resample(X_train_scaled, y_train)
    print(f"[INFO] Size of train after undersampling: {X_train_bal.shape}")
    print(
        f"[INFO] Classes in train after undersampling:\n{pd.Series(y_train_bal).value_counts()}"
    )

    return X_train_bal, X_test_scaled, y_train_bal, y_test, scaler


def catboost_cv(X, y, iterations, learning_rate, depth, l2_leaf_reg, folds=3):
    """
    Cross validation CatBoost with given hyperparameters.
    Returns average ROC-AUC on validation.
    """
    params = {
        "iterations": int(iterations),
        "learning_rate": learning_rate,
        "depth": int(depth),
        "l2_leaf_reg": l2_leaf_reg,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": False,
        "scale_pos_weight": (len(y) - sum(y)) / sum(y),
    }

    pool = Pool(X, y)
    cv_results = cv(
        pool=pool,
        params=params,
        fold_count=folds,
        stratified=True,
        shuffle=True,
        partition_random_seed=42,
        early_stopping_rounds=50,
        verbose=False,
    )

    best_auc = cv_results["test-AUC-mean"].max()
    return best_auc


def bayesian_optimization_cv(X, y):
    """
    Searching optimal learning_rate and other parameters using Bayesian Optimization
    """

    def cv_func(iterations, learning_rate, depth, l2_leaf_reg):
        auc = catboost_cv(
            X,
            y,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            folds=3,
        )
        print(
            f"[BayesOpt] iter={int(iterations)}, lr={learning_rate:.5f}, depth={int(depth)}, l2={l2_leaf_reg:.3f} => AUC={auc:.5f}"
        )
        return auc

    optimizer = BayesianOptimization(
        f=cv_func,
        pbounds={
            "iterations": (300, 700),
            "learning_rate": (0.01, 0.07),
            "depth": (4, 9),
            "l2_leaf_reg": (5, 15),
        },
        random_state=42,
        verbose=2,
    )
    optimizer.maximize(init_points=3, n_iter=10)

    best_params = optimizer.max["params"]
    best_params["iterations"] = int(best_params["iterations"])
    best_params["depth"] = int(best_params["depth"])
    best_params["learning_rate"] = float(best_params["learning_rate"])
    best_params["l2_leaf_reg"] = float(best_params["l2_leaf_reg"])
    print(f"[BayesOpt] Best parameters: {best_params}")
    return best_params


# def train_catboost(X_train, y_train, params):
def train_catboost(X_train, y_train, X_test_scaled, y_test, params):
    """
    Studying CatBoost with given parameters
    """
    print("[INFO] Studying final model with given parameters:")
    print(params)
    model = CatBoostClassifier(
        **params, early_stopping_rounds=50, verbose=100, random_seed=42
    )
    # model.fit(X_train, y_train, use_best_model=True)
    model.fit(X_train, y_train, eval_set=(X_test_scaled, y_test), use_best_model=True)
    return model


def evaluate_model(model, X_test, y_test, target_recall=0.7):
    """
    Model Evaluation  with threshold tuning: maximize Precision at Recall >= target_recall.
    """
    print("[INFO] Assess model on testing selection...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Select threshold: maximise Precision with Recall >= target_recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    possible = [
        (p, r, t)
        for p, r, t in zip(precision, recall, thresholds)
        if r >= target_recall
    ]
    if possible:
        best_p, best_r, best_t = max(possible, key=lambda x: x[0])
        print(
            f"[INFO] Best threshold for max Precision at Recall>={target_recall}: {best_t:.4f}, Precision: {best_p:.4f}, Recall: {best_r:.4f}"
        )
        y_pred_opt = (y_pred_proba >= best_t).astype(int)
    else:
        print(
            f"[INFO] No threshold found for Recall >= {target_recall}. Using default 0.5"
        )
        best_t = 0.5
        y_pred_opt = (y_pred_proba >= best_t).astype(int)
        best_p = precision_score(y_test, y_pred_opt)
        best_r = recall_score(y_test, y_pred_opt)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_opt)

    print(f"[RESULT] ROC-AUC: {roc_auc:.4f}")
    print(f"[RESULT] F1-Score (optimal threshold): {f1:.4f}")
    print(f"[RESULT] Precision (optimal threshold): {best_p:.4f}")
    print(f"[RESULT] Recall (optimal threshold): {best_r:.4f}")

    plot_all_metrics(y_test, y_pred_proba, y_pred_opt, best_t, best_p, best_r)


def plot_all_metrics(y_true, y_scores, y_pred, best_t, best_p, best_r):
    """
    Plot Precision-Recall curve, ROC curve and Confusion Matrix
    in a single figure with 3 subplots. Mark optimal threshold on PR curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Precision-Recall curve
    axs[0].plot(recall, precision, label=f"AP = {ap:.4f}")
    axs[0].scatter(
        [best_r],
        [best_p],
        color="red",
        label=f"Best threshold\nRecall={best_r:.2f}\nPrecision={best_p:.2f}",
    )
    axs[0].set_xlabel("Recall")
    axs[0].set_ylabel("Precision")
    axs[0].set_title("Precision-Recall Curve")
    axs[0].legend()
    axs[0].grid(True)

    # ROC curve
    axs[1].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    axs[1].plot([0, 1], [0, 1], "k--")
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate (Recall)")
    axs[1].set_title("ROC Curve")
    axs[1].legend()
    axs[1].grid(True)

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[2])
    axs[2].set_xlabel("Predicted")
    axs[2].set_ylabel("Actual")
    axs[2].set_title("Confusion Matrix\n(optimal threshold)")

    plt.tight_layout()
    plt.show()


def main():
    data_path = "creditcard.csv"

    df = load_data(data_path)

    # Preprocessing and balancing
    X_train_bal, X_test_scaled, y_train_bal, y_test, scaler = preprocess_and_split(df)

    # Searching optimal hyperparameters (hyperparametric optimization) using Bayesian Optimization for training data only
    best_params = bayesian_optimization_cv(X_train_bal, y_train_bal)

    # Train final model with given parameters on balancing train
    model = train_catboost(X_train_bal, y_train_bal, X_test_scaled, y_test, best_params)

    # Evaluate model for non balancing train with given parameters
    evaluate_model(model, X_test_scaled, y_test)

    joblib.dump(model, "catboost_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("[INFO] Модель and scaler are saved.")


if __name__ == "__main__":
    main()
