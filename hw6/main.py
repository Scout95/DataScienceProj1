import os
PLOTS_DIR = "hw6_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Без GUI, только сохранение графиков
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, classification_report
from data_loader import DataLoader
import umap
from scipy.stats import mode
import tensorflow as tf
keras = tf.keras
from keras.layers import Input, Dense
from keras.models import Model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Exited at iteration.*")
warnings.filterwarnings("ignore", message=".*spectral initialisation failed.*")
warnings.filterwarnings("ignore", message=".*Exited postprocessing with accuracies.*")

def savefig(name):
    plt.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight')
    plt.close()


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
        print("There are heading and first 5 rows of table:")
        print(df.head())
    except RuntimeError as e:
        print(e)
        return

    # df = pd.read_csv(path)
    return df


df = load_data("./hw6/dataset/customer_shopping_data.csv")
if df is None:
    raise SystemExit("Data loading failed, exiting.")

# --- Проверка пропусков ---
print("\n[STEP] Checking for missing values...")
print("\nMissing values per column:")
print(df.isnull().sum())

print("Размер датасета:", df.shape)
print("\nData info:")
print(df.info())
print("\nData columns:")
print(df.columns)

# --- 2. Предварительная обработка данных ---
# Приведение колонок к нижнему регистру для удобства
df.columns = df.columns.str.lower()

# Преобразование даты в datetime
df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')

# Проверка пропусков
print("Пропуски по колонкам:\n", df.isnull().sum())

# --- Заполнение пропусков: числовые - медианой, категориальные - модой ---
num_cols = ['age', 'quantity', 'price']
cat_cols = ['gender', 'category', 'payment_method', 'shopping_mall']

print("\n[STEP] Imputing missing number values with median...")
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])
# data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

print("\n[STEP] Imputing missing categorian values with most_frequent...")
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Проверка пропусков еще раз
print("Пропуски после заполнения:\n", df.isnull().sum())

# Создание нового признака - total_purchase_amount
df['total_purchase_amount'] = df['quantity'] * df['price']

# Кодирование категориальных переменных (пол)
le_gender = LabelEncoder()
df['gender_enc'] = le_gender.fit_transform(df['gender'])

# Сохранение копии исходного DataFrame для EDA
df_eda = df.copy()

# One-Hot Encoding для категорий и способов оплаты
df = pd.get_dummies(df, columns=['category', 'payment_method', 'shopping_mall'], drop_first=True)


# --- Создание признаков из даты с учетом пропусков ---
df['purchase_month'] = np.where(
    df['invoice_date'].notnull(),
    df['invoice_date'].dt.month,
    -1
)
df['purchase_dayofweek'] = np.where(
    df['invoice_date'].notnull(),
    df['invoice_date'].dt.dayofweek,
    -1
)

# Преобразование purchase_month и purchase_dayofweek в строковый тип
df['purchase_month'] = df['purchase_month'].astype(str)
df['purchase_dayofweek'] = df['purchase_dayofweek'].astype(str)

# One-Hot Encoding для категориальных признаков включая purchase_month и purchase_dayofweek
# df = pd.get_dummies(
#     df,
#     columns=['category', 'payment_method', 'shopping_mall', 'purchase_month', 'purchase_dayofweek'],
#     drop_first=True
# )
df = pd.get_dummies(
    df,
    columns=['purchase_month', 'purchase_dayofweek'],
    drop_first=True
)

# # Проверка
# print(df[['invoice_date', 'purchase_month', 'purchase_dayofweek']].head(10))
# print("Пример после обработки:")
# print(df.head())
# print("Значения purchase_month:", df['purchase_month'].unique())
# print("Значения purchase_dayofweek:", df['purchase_dayofweek'].unique())


# --- 2.1 Формирование (выбор) признаков для кластеризации ---
# features = ['age', 'gender_enc', 'total_purchase_amount', 'quantity', 'purchase_month', 'purchase_dayofweek']
features = ['age', 'gender_enc', 'total_purchase_amount', 'quantity'] + \
           [col for col in df.columns if col.startswith('category_') or
            col.startswith('payment_method_') or col.startswith('shopping_mall_') or
            col.startswith('purchase_month_') or col.startswith('purchase_dayofweek_')]

# --- Формирование списка признаков для обучения ---
# features = [col for col in df.columns if col not in ['invoice_date', 'gender', 'shopping_mall', 'category', 'payment_method']]

X = df[features]

# --- Проверка ---
print("NaN в X перед масштабированием:\n", X.isnull().sum())

# !!! ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА И ЗАПОЛНЕНИЕ ПРОПУСКОВ !!!
df[features] = df[features].fillna(df[features].median())

print("NaN после всех преобразований:", df[features].isnull().sum())

print(X.isnull().sum())
print(np.isinf(X).sum())


print("NaN в X перед масштабированием:\n", X.isnull().sum())
print("inf в X перед масштабированием:\n", np.isinf(X).sum())
print("min/max по признакам:\n", X.min(), X.max())


# --- 2.2 Масштабирование ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("NaN in X_scaled:", np.isnan(X_scaled).sum())


# --- Проверка уникальности и NaN ---
print("X_scaled shape:", X_scaled.shape)
print("Уникальных строк:", np.unique(X_scaled, axis=0).shape[0])
print("NaN in X_scaled:", np.isnan(X_scaled).sum())
# -------------------------

# --- Автоэнкодер ---

input_dim = X_scaled.shape[1]
encoding_dim = 10  # размер латентного пространства, имеет смысл подбирать

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Обучение автоэнкодера с сохранением истории ---
history = autoencoder.fit(X_scaled, X_scaled,
                          epochs=50,
                          batch_size=32,
                          shuffle=True,
                          validation_split=0.1,
                          verbose=1)
# autoencoder.fit(X_scaled, X_scaled,
#                 epochs=50,
#                 batch_size=32,
#                 shuffle=True,
#                 validation_split=0.1,
#                 verbose=1)

# ----

# --- График ошибки (loss) по эпохам ---
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Ошибка восстановления автоэнкодера по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
# plt.show()
savefig("Ошибка восстановления автоэнкодера по эпохам.png")
plt.close()


# --- Визуализация примера восстановления ---
reconstructed = autoencoder.predict(X_scaled)

# Выбираем случайный пример для визуализации
idx = np.random.randint(0, X_scaled.shape[0])

plt.figure(figsize=(10,4))
plt.plot(X_scaled[idx], label='Оригинал')
plt.plot(reconstructed[idx], label='Восстановлено', linestyle='--')
plt.title(f'Пример восстановления входного вектора (индекс {idx})')
plt.xlabel('Признак')
plt.ylabel('Значение (масштабированное)')
plt.legend()
# plt.show()
savefig("Пример восстановления входного вектора.png")
plt.close()


# --- Кластеризация латентных признаков ---
encoder = Model(inputs=input_layer, outputs=encoded)
X_latent = encoder.predict(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_latent)

df['deep_cluster'] = cluster_labels

print("Распределение по сегментам Deep Learning кластеризации:")
print(df['deep_cluster'].value_counts())


# ----


# Получение латентных признаков
encoder = Model(inputs=input_layer, outputs=encoded)
X_latent = encoder.predict(X_scaled)

# Кластеризация в латентном пространстве
kmeans = KMeans(n_clusters=5, random_state=42)
deep_cluster_labels = kmeans.fit_predict(X_latent)

# Добавление меток сегментов в DataFrame
df['deep_cluster'] = deep_cluster_labels

# --- Итоги ---
print("Распределение по сегментам Deep Learning кластеризации:")
print(df['deep_cluster'].value_counts())

# Сохраняем результат
df.to_csv("customer_data_with_deep_clusters.csv", index=False)


# 1) Снижение размерности для визуализации
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
X_umap = reducer.fit_transform(X_latent)

# 2) Визуализация сегментов
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
scatter = plt.scatter(X_umap[:,0], X_umap[:,1], c=deep_cluster_labels, cmap='tab10', s=10)
plt.colorbar(scatter, label='Cluster label')
plt.title('UMAP визуализация сегментов автоэнкодера')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
# plt.show()
savefig("UMAP визуализация сегментов автоэнкодера")
plt.close()




# ----------------------------
# Добавляем небольшой шум
noise_level = 1e-5
X_noisy = X_scaled + noise_level * np.random.normal(size=X_scaled.shape)


# --- 2.3 Понижение размерности с помощью UMAP ---
# reducer = umap.UMAP(n_components=2, random_state=42)
# X_umap = reducer.fit_transform(X_scaled)
# Запускаем UMAP с увеличенным числом соседей и случайной инициализацией
# reducer = umap.UMAP(n_neighbors=50, init='random', random_state=42)

# Инициализация UMAP с увеличенным числом соседей, случайной инициализацией и параллелизмом
reducer = umap.UMAP(
    n_neighbors=50,    # увеличенное число соседей
    init='random',     # случайная инициализация
    n_jobs=-1          # использовать все доступные ядра для параллелизма
    # random_state не задаём, чтобы параллелизм работал
)

X_umap = reducer.fit_transform(X_noisy)

# reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
# # reducer = umap.UMAP(n_components=2, n_jobs=-1)  # Использует все ядра, без фиксации random_state
# X_umap = reducer.fit_transform(X_scaled)

# --- 2.4 Кластеризация KMeans на UMAP-пространстве ---
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_umap)

# --- 2.5 Оценка качества кластеризации ---
score = silhouette_score(X_umap, kmeans_labels)
print(f"Silhouette Score (UMAP + KMeans): {score:.3f}")

# --- 2.6 Добавление меток кластеров в DataFrame ---
df['cluster'] = kmeans_labels



# --- 2.7 Анализ аномалий ---
# Используем Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['iso_anomaly'] = iso_forest.fit_predict(X_scaled)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
df['lof_anomaly'] = lof.fit_predict(X_scaled)

# One-Class SVM
ocsvm = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
ocsvm.fit(X_scaled)
df['ocsvm_anomaly'] = ocsvm.predict(X_scaled)

print("Аномалии по методам:")
print("Isolation Forest:", (df['iso_anomaly'] == -1).sum())
print("LOF:", (df['lof_anomaly'] == -1).sum())
print("One-Class SVM:", (df['ocsvm_anomaly'] == -1).sum())


# --- 2.8 Визуализация кластеров ---
plt.figure(figsize=(10,7))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=kmeans_labels, palette='Set1', legend='full')
plt.title('Кластеры KMeans на UMAP-пространстве')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
# plt.show()
savefig("Кластеры KMeans на UMAP-пространстве")
plt.close()


# --- 2.9 Визуализация аномалий (Isolation Forest) ---
plt.figure(figsize=(10,7))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=df['iso_anomaly'], palette={1:'blue', -1:'red'})
plt.title('Аномалии Isolation Forest (красным)')
# plt.show()
savefig("Аномалии Isolation Forest (красным)")
plt.close()








# --- 2.10 Анализ сегментов ---
print(df.groupby('cluster')[features].mean())

# --- 2.11. Сохранение результатов ---
df.to_csv('customer_shopping_segmented.csv', index=False)
print("Результаты сегментации и аномалий сохранены в 'customer_shopping_segmented.csv'")


# --- 3. Exploratory Data Analysis (EDA) ---
print("Exploratory Data Analysis (EDA)")

plt.figure(figsize=(12,5))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Распределение возраста покупателей')
# plt.show()
savefig("Распределение возраста покупателей")
plt.close()


plt.figure(figsize=(12,5))
# sns.boxplot(x='gender', y='total_purchase_amount', data=df)
sns.boxplot(x='gender', y='total_purchase_amount', data=df_eda)
plt.title('Сумма покупки по полу')
# plt.show()
savefig("Сумма покупки по полу")

# После One-Hot Encoding исходная колонка 'payment_method' исчезла, а dummy-колонки содержат только 0/1.
# Для countplot по категориям восстановление категориального признака
payment_cols = [col for col in df.columns if col.startswith('payment_method_')]
df['payment_method_reconstructed'] = (
    df[payment_cols].idxmax(axis=1).str.replace('payment_method_', '')
)
plt.figure(figsize=(12,5))
print(f"Колонки в df: '{df.columns}'")
# sns.countplot(x='payment_method_Credit Card', data=df)
sns.countplot(x='payment_method_reconstructed', data=df)
plt.title('Распределение способов оплаты')
# plt.show()
savefig("Распределение способов оплаты")
plt.close()


plt.figure(figsize=(12,5))
# sns.scatterplot(x='age', y='total_purchase_amount', hue='gender', data=df)
sns.scatterplot(x='age', y='total_purchase_amount', hue='gender', data=df_eda)
plt.title('Зависимость суммы покупки от возраста и пола')
# plt.show()
savefig("Зависимость суммы покупки от возраста и пола")
plt.close()


# Корреляция числовых признаков
# plt.figure(figsize=(10,8))
# sns.heatmap(df[['age', 'quantity', 'price', 'total_purchase_amount', 'purchase_month', 'purchase_dayofweek', 'gender_enc']].corr(), annot=True, cmap='coolwarm')
# plt.title('Корреляционная матрица')
# plt.xticks(rotation=45, ha='right', fontsize=10)
# plt.subplots_adjust(bottom=0.35, left=0.25)
# plt.tight_layout() 
# # plt.show()
# savefig("Корреляционная матрица")
# plt.close()

plt.figure(figsize=(10,8))
corr_cols = ['age', 'quantity', 'price', 'total_purchase_amount', 'gender_enc']
corr_cols += [col for col in df.columns if col.startswith('purchase_month_') or col.startswith('purchase_dayofweek_')]
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.subplots_adjust(bottom=0.35, left=0.25)
plt.tight_layout()
savefig("Корреляционная матрица")
plt.close()



# # 4.2 Agglomerative Clustering
# agglo = AgglomerativeClustering(n_clusters=5)
# agglo_labels = agglo.fit_predict(X_scaled)

# 4.3 DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 4.4 Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)









# Визуализация кластеров KMeans на PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,7))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=kmeans_labels, palette='Set1', legend='full')
plt.title('KMeans кластеры на PCA проекции')
# plt.show()
savefig("KMeans кластеры на PCA проекции")
plt.close()



# --- 7. Важность признаков для предсказания кластера (пример RandomForest на KMeans) ---
print("Важность признаков для предсказания кластера (пример RandomForest на KMeans)")
rf = RandomForestClassifier(random_state=42)
rf.fit(X, kmeans_labels)
importances = rf.feature_importances_

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('Важность признаков для кластеров KMeans')
# plt.show()
savefig("Важность признаков для кластеров KMeans")
plt.close()


# --- 8. Графики ошибок и метрик ---

# Silhouette score для всех моделей
print("Silhouette Scores:")
print(f"KMeans: {silhouette_score(X_scaled, kmeans_labels):.3f}")
# print(f"Agglomerative: {silhouette_score(X_scaled, agglo_labels):.3f}")
print(f"GMM: {silhouette_score(X_scaled, gmm_labels):.3f}")
# print(f"Spectral: {silhouette_score(X_scaled, spectral_labels):.3f}")

# DBSCAN может иметь шум, поэтому подсчет только по ненулевым кластерам
mask = dbscan_labels != -1
if mask.sum() > 0:
    print(f"DBSCAN: {silhouette_score(X_scaled[mask], dbscan_labels[mask]):.3f}")
else:
    print("DBSCAN: Нет кластеров для оценки")


# --- АНСАМБЛЕВЫЕ ПОДХОДЫ ДЛЯ КЛАСТЕРИЗАЦИИ ---
# после получения всех меток кластеризации: kmeans_labels, agglo_labels, dbscan_labels, gmm_labels


# Корректная обработка меток DBSCAN: -1 (шум) заменяем на отдельный кластер
# Для DBSCAN -1 (шум) заменим на отдельный кластер
dbscan_labels_adj = dbscan_labels.copy()
if (dbscan_labels_adj == -1).any():
    dbscan_labels_adj[dbscan_labels_adj == -1] = dbscan_labels_adj.max() + 1

# Собираем все метки в матрицу (n_samples, n_models)
# Формируем матрицу меток (n_samples, n_models)
cluster_labels_matrix = np.vstack([
    kmeans_labels,
    # agglo_labels,
    dbscan_labels_adj,
    gmm_labels
]).T  # shape: (n_samples, n_models)

# --- 1. Consensus Clustering (Majority Vote) ---
# Итоговая метка — наиболее частая среди всех моделей
ensemble_labels, _ = mode(cluster_labels_matrix, axis=1)
df['ensemble_cluster_majority'] = ensemble_labels.flatten()

print("\n[Consensus Clustering] Распределение по кластерам (majority vote):")
print(df['ensemble_cluster_majority'].value_counts())

plt.figure(figsize=(10,7))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=df['ensemble_cluster_majority'], palette='Set2', legend='full')
plt.title('Consensus Clustering (Majority Vote) на UMAP-пространстве')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
# plt.show()
savefig("Consensus Clustering (Majority Vote) на UMAP-пространстве")
plt.close()


if len(np.unique(df['ensemble_cluster_majority'])) > 1:
    print(f"Silhouette Score (Consensus Majority): {silhouette_score(X_umap, df['ensemble_cluster_majority']):.3f}")
else:
    print("Consensus Majority: только один кластер, silhouette score не вычисляется.")

# # --- 2. Co-association Matrix Ensemble Clustering ---
# # Для каждой пары объектов считаем, сколько раз они попали в один кластер в разных моделях
# print("\n[Co-association Matrix Ensemble Clustering]")

# n_samples = cluster_labels_matrix.shape[0]
# n_models = cluster_labels_matrix.shape[1]
# coassoc = np.zeros((n_samples, n_samples), dtype=float)

# for m in range(n_models):
#     labels = cluster_labels_matrix[:, m]
#     for i in range(n_samples):
#         for j in range(i, n_samples):
#             if labels[i] == labels[j]:
#                 coassoc[i, j] += 1
#                 if i != j:
#                     coassoc[j, i] += 1
# coassoc /= n_models

# from sklearn.cluster import AgglomerativeClustering
# n_clusters = len(np.unique(kmeans_labels))
# agglo_coassoc = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
# coassoc_labels = agglo_coassoc.fit_predict(1 - coassoc)
# df['ensemble_cluster_coassoc'] = coassoc_labels

# print("Распределение по кластерам (co-association):")
# print(df['ensemble_cluster_coassoc'].value_counts())

# plt.figure(figsize=(10,7))
# sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=df['ensemble_cluster_coassoc'], palette='Set2', legend='full')
# plt.title('Ensemble Clustering (Co-association Matrix) на UMAP-пространстве')
# plt.xlabel('UMAP-1')
# plt.ylabel('UMAP-2')
# # plt.show()
# savefig("Ensemble Clustering (Co-association Matrix) на UMAP-пространстве")
# plt.close()


# if len(np.unique(df['ensemble_cluster_coassoc'])) > 1:
#     print(f"Silhouette Score (Co-association): {silhouette_score(X_umap, df['ensemble_cluster_coassoc']):.3f}")
# else:
#     print("Co-association: только один кластер, silhouette score не вычисляется.")

# # --- 3. Soft Voting (Probabilistic Assignment via GMM) ---
# # Используем вероятности принадлежности к кластерам из GMM и выбираем наиболее вероятный кластер
# print("\n[Soft Voting/Probabilistic Assignment via GMM]")

# gmm_probs = gmm.predict_proba(X_scaled)
# soft_labels = np.argmax(gmm_probs, axis=1)
# df['ensemble_cluster_soft_gmm'] = soft_labels

# print("Распределение по кластерам (soft voting GMM):")
# print(df['ensemble_cluster_soft_gmm'].value_counts())

# plt.figure(figsize=(10,7))
# sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=df['ensemble_cluster_soft_gmm'], palette='Set2', legend='full')
# plt.title('Soft Voting (GMM Probabilities) на UMAP-пространстве')
# plt.xlabel('UMAP-1')
# plt.ylabel('UMAP-2')
# # plt.show()
# savefig("Soft Voting (GMM Probabilities) на UMAP-пространстве")
# plt.close()


# if len(np.unique(df['ensemble_cluster_soft_gmm'])) > 1:
#     print(f"Silhouette Score (Soft Voting GMM): {silhouette_score(X_umap, df['ensemble_cluster_soft_gmm']):.3f}")
# else:
#     print("Soft Voting GMM: только один кластер, silhouette score не вычисляется.")

# # --- 4. Consensus Functions from sklearn-extra ---
# # Использует специальные алгоритмы (CSPA, HGPA, MCLA) для объединения разметок кластеров
# print("\n[Consensus Functions from sklearn-extra]")

# try:
#     from sklearn_extra.cluster import cluster_ensembles

#     # cluster_ensembles принимает метки в формате (n_samples, n_models)
#     # и возвращает итоговые метки
#     consensus_labels = cluster_ensembles(
#         cluster_labels_matrix, verbose=True, solver='cspa', nclass=n_clusters
#     )
#     df['ensemble_cluster_sklearn_extra'] = consensus_labels

#     print("Распределение по кластерам (sklearn-extra consensus):")
#     print(df['ensemble_cluster_sklearn_extra'].value_counts())

#     plt.figure(figsize=(10,7))
#     sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=df['ensemble_cluster_sklearn_extra'], palette='Set2', legend='full')
#     plt.title('Consensus Clustering (sklearn-extra) на UMAP-пространстве')
#     plt.xlabel('UMAP-1')
#     plt.ylabel('UMAP-2')
#     # plt.show()
#     savefig("Consensus Clustering (sklearn-extra) на UMAP-пространстве")
#     plt.close()


#     if len(np.unique(df['ensemble_cluster_sklearn_extra'])) > 1:
#         print(f"Silhouette Score (sklearn-extra consensus): {silhouette_score(X_umap, df['ensemble_cluster_sklearn_extra']):.3f}")
#     else:
#         print("sklearn-extra consensus: только один кластер, silhouette score не вычисляется.")
# except ImportError:
#     print("sklearn-extra не установлен. Установите через 'pip install scikit-learn-extra' для использования consensus functions.")




# --- Итоговый вывод ---
print("""
Анализ завершен. Данные очищены, категориальные признаки преобразованы.
Проведена сегментация клиентов 5 методами.
Выявлены аномалии тремя методами.
Визуализированы распределения, кластеры и важность признаков.
Возможно использовать сегменты для таргетинга и анализ аномалий для контроля качества.
""")



# # from IPython.display import display
# # display(results_df_sorted)
# # For Jupyter Notebook:
# # display(results_df)
# # try:
# #     from IPython.display import display
# #     display(results_df_sorted)
# # except ImportError:
# #     pass
