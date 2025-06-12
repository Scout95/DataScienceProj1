from matplotlib import pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from data_loader import DataLoader
import seaborn as sns
import os
from pathlib import Path

from sklearn.pipeline import make_pipeline


def load_data(path: str) -> pd.DataFrame:
    loader = DataLoader()
    base_dir = Path.cwd()  # or Path(__file__).parent

    # # Load data from csv
    current_dir = os.getcwd()
    path_to_file = os.path.join(base_dir, path)

    try:
        # print(f"[INFO] Loading data from file: {path_to_file}")
        df = loader.load_csv(path_to_file)
        print(f"[INFO] Data successfully loaded. Size: {df.shape}")
        print(df["csMPa"].value_counts())
        print("There are heading and first 5 rows of table:")
        print(df.head())
    except RuntimeError as e:
        print(e)
        return

    # df = pd.read_csv(path)
    return df


data = load_data("./hw5/dataset/concrete_data.csv")
if data is None:
    raise SystemExit("Data loading failed, exiting.")

# --- Проверка пропусков ---
print("\n[STEP] Checking for missing values...")
print("\nMissing values per column:")
print(data.isnull().sum())

print("\nData info:")
print(data.info())
print("\nData columns:")
print(data.columns)


# --- Заполнение пропусков медианой ---
print("\n[STEP] Imputing missing values with median...")
imputer = SimpleImputer(strategy="median")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# --- Визуализация выбросов с помощью boxplot ---
print("\n[STEP] Visualizing outliers with boxplot...")
plt.figure(figsize=(12, 8))
sns.boxplot(data=data_imputed)
plt.title("Boxplot для признаков (поиск выбросов)")
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.25) 
plt.show()
print("[INFO] Boxplot visualization completed.")

# --- Обработка выбросов ---
# Вместо удаления выбросов используем RobustScaler, который устойчив к ним
#Для фильтрования выбросов - раскомментировать следующий блок:

# Q1 = data_imputed.quantile(0.25)
# Q3 = data_imputed.quantile(0.75)
# IQR = Q3 - Q1
# mask = ~((data_imputed < (Q1 - 1.5 * IQR)) | (data_imputed > (Q3 + 1.5 * IQR))).any(axis=1)
# data_filtered = data_imputed[mask]
# print(f"Размер данных до фильтрации выбросов: {data_imputed.shape}")
# print(f"Размер данных после фильтрации выбросов: {data_filtered.shape}")
# data_to_use = data_filtered

data_to_use = data_imputed  # используем все данные с заполненными пропусками
# --- Data preparation: Разделение признаков и целевой переменной ---
target_column = "csMPa"
if target_column not in data_to_use.columns:
    raise KeyError(f"Целевая переменная '{target_column}' отсутствует в данных.")

X = data_to_use.drop(target_column, axis=1)
y = data_to_use[target_column]

# --- Разделение на обучающую и тестовую выборки ---
# stratify не используется для регрессии
print("\n[STEP] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
# ------------
# --- Масштабирование с сохранением DataFrame ---
print("\n[STEP] Scaling features with RobustScaler...")
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
print("[INFO] Scaling completed.")

# # --- Масштабирование признаков отдельно ---
# scaler = RobustScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# --- Инициализация ,base моделей ---
print("\n[STEP] Initializing models...")
models = {
    "PassiveAggressive": make_pipeline(RobustScaler(), PassiveAggressiveRegressor(max_iter=1000, random_state=42)),
    "KNeighbors": make_pipeline(RobustScaler(), KNeighborsRegressor()),
    "ElasticNet": make_pipeline(RobustScaler(), ElasticNet(random_state=42)),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "LightGBM": LGBMRegressor(
        random_state=42,
        force_row_wise=True,
        n_estimators=100,
        num_leaves=31,
        min_data_in_leaf=5,
        learning_rate=0.05,
        verbosity=-1,
    ),
    "Dummy": DummyRegressor(strategy="mean"),
}
print("[INFO] Models initialized.")

# # Для моделей, чувствительных к масштабу, используем Pipeline с RobustScaler
# models = {
#     "PassiveAggressive": make_pipeline(RobustScaler(), PassiveAggressiveRegressor(max_iter=1000, random_state=42)),
#     "KNeighbors": make_pipeline(RobustScaler(), KNeighborsRegressor()),
#     "ElasticNet": make_pipeline(RobustScaler(), ElasticNet(random_state=42)),
#     "DecisionTree": DecisionTreeRegressor(random_state=42), # деревья не требуют масштабирования
#     "RandomForest": RandomForestRegressor(random_state=42), # деревья не требуют масштабирования
#     "LightGBM": make_pipeline(RobustScaler(), LGBMRegressor(random_state=42, force_row_wise=True)),
#     "Dummy": DummyRegressor(strategy="mean"),
# }

# LightGBM обучаем отдельно на numpy-массивах (масштабированных)
# lgbm_model = LGBMRegressor(
#     random_state=42,
#     force_row_wise=True,
#     n_estimators=100,
#     num_leaves=31,
#     min_data_in_leaf=5,
#     learning_rate=0.05,
#     verbosity=-1,
# )
# models["LightGBM"] = lgbm_model


# --- Обучение моделей и получение предсказаний ---
print("\n[STEP] Training models and making predictions...")
results = {}
predictions = {}

for name, model in models.items():
    print(f"[INFO] Training model: {name}...")
    if name == "LightGBM":
        # Обучаем и предсказываем на масштабированных numpy-массивах
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Для остальных моделей используем DataFrame (Pipeline внутри)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    predictions[name] = y_pred
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "MAE": mae, "R2": r2}
    print(f"[RESULT] Model '{name}' trained. R²: {r2:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}")

# --- Создание ансамбля VotingRegressor на основе нескольких моделей 
#  с передачей DataFrame с именами признаков---
# Используем три модели с разными подходами
print("\n[STEP] Training VotingRegressor ensemble...")
voting_regressor = VotingRegressor([
    ('rf', models['RandomForest']),
    # ('en', models['ElasticNet']),
    # ('knn', models['KNeighbors'])
    ('lgbm', models['LightGBM']),
    ('knn', models['KNeighbors'])
])

# Для ансамбля масштабируем данные отдельно на масштабированных DataFrame
voting_regressor.fit(X_train_scaled, y_train)
y_pred_voting = voting_regressor.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred_voting)
mae = mean_absolute_error(y_test, y_pred_voting)
r2 = r2_score(y_test, y_pred_voting)
results['VotingRegressor'] = {"MSE": mse, "MAE": mae, "R2": r2}
predictions['VotingRegressor'] = y_pred_voting
print(f"[RESULT] Model 'VotingRegressor' trained. R²: {r2:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}")


# --- Таблица результатов ---
results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
results_df = results_df.round({"R2": 3, "MSE": 3, "MAE": 3})

print("\n[STEP] Evaluation results (unsorted):")
print(results_df)

# --- Сортировка и вывод результатов ---
results_df_sorted = results_df.sort_values(
    by=['R2', 'MSE', 'MAE'],
    ascending=[False, True, True]
).reset_index(drop=True)

print("\nEvaluation results (sorted):")
print("\n[STEP] Evaluation results (sorted):")
print(results_df_sorted)

# --- Визуализация метрик ---
print("\n[STEP] Visualizing metrics...")
plt.figure(figsize=(15, 5))
metrics = ["R2", "MSE", "MAE"]

for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    sns.barplot(x="Model", y=metric, hue="Model", data=results_df, palette="viridis", legend=False)
    plt.title(metric)
    plt.xticks(rotation=45)
    if metric == "R2":
        plt.ylim(0, 1)
plt.tight_layout()
plt.show()
print("[INFO] Metrics visualization completed.")


# --- Визуализация предсказаний лучшей модели ---
best_model_name = results_df.sort_values(by="R2", ascending=False).iloc[0]["Model"]
print(f"\n[STEP] Visualizing predictions for the best model by R²: {best_model_name}")

# Получаем сам регрессор (класс) для лучшей модели
if best_model_name == "VotingRegressor":
    regressor_type = type(voting_regressor).__name__
else:
    regressor_type = type(models[best_model_name]).__name__
print(f"Regressor class for best model: {regressor_type}")

y_pred_best = predictions[best_model_name]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.title(f"Предсказания модели '{best_model_name}'")
plt.show()
print("[INFO] Best model prediction visualization completed.")



# models['VotingRegressor'] = voting_regressor

# # --- Обучение моделей и получение предсказаний ---
# results = {}
# predictions = {}

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     predictions[name] = y_pred
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     results[name] = {"MSE": mse, "MAE": mae, "R2": r2}
#     print(f"Model '{name}' trained. R²: {r2:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}")

# # --- Создание DataFrame (results table) с результатами ---
# results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
# results_df["R2"] = results_df["R2"].round(3)
# results_df["MSE"] = results_df["MSE"].round(3)
# results_df["MAE"] = results_df["MAE"].round(3)
# {"index": "Model"})
# results_df = results_df.round({"R2": 3, "MSE": 3, "MAE": 3})

# print("\nEvaluation results:")
# print(results_df)

# # --- Визуализация метрик ---
# plt.figure(figsize=(15, 5))
# metrics = ["R2", "MSE", "MAE"]

# for i, metric in enumerate(metrics, 1):
#     plt.subplot(1, 3, i)
#     sns.barplot(x="Model", y=metric, hue="Model", data=results_df, palette="viridis", legend=False)
#     plt.title(metric)
#     plt.xticks(rotation=45)
#     if metric == "R2":
#         plt.ylim(0, 1)
# plt.tight_layout()
# plt.show()

# # --- Визуализация предсказаний лучшей модели ---
# best_model_name = results_df.sort_values(by="R2", ascending=False).iloc[0]["Model"]
# print(f"\nBest model by R²: {best_model_name}")

# y_pred_best = predictions[best_model_name]

# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred_best, alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
# plt.xlabel("Истинные значения")
# plt.ylabel("Предсказанные значения")
# plt.title(f"Предсказания модели '{best_model_name}'")
# plt.show()

# from IPython.display import display
# display(results_df_sorted)
# For Jupyter Notebook:
# display(results_df)
# try:
#     from IPython.display import display
#     display(results_df_sorted)
# except ImportError:
#     pass
