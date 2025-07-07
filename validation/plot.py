import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from train import (
    BOTTLENECK_DIM,
    Autoencoder,
    CustomDataset,
    load_data,
    FRAME_SIZE,
)

SEED = 42

data = pd.read_csv("full_latents.csv")

data = data.dropna()
features_l = ["l" + str(i) for i in range(BOTTLENECK_DIM)]

# ===== НАСТРОЙКИ =====
class_multipliers = {
    0: 2,  # Для 0-го класса в 2 раза больше, чем min_class_count
    1: 1,   # Для 1-го стандартное количество
    2: 1,   # Для 2-го стандартное количество
    3: 1,   # Для 3-го стандартное количество
}
# =====================

# Находим минимальный размер среди всех классов (или только указанных)
valid_classes = [cls for cls in class_multipliers if cls in data["target"].unique()]
if not valid_classes:
    raise ValueError("Нет указанных классов в данных!")

min_class_count = min(
    len(data[data["target"] == cls]) // class_multipliers[cls] 
    for cls in valid_classes
)

# Собираем данные в соответствии с множителями
sampled_data = []
for class_label, multiplier in class_multipliers.items():
    if class_label not in data["target"].unique():
        continue  # Пропускаем отсутствующие классы
    
    n_samples = min_class_count * multiplier
    class_data = data[data["target"] == class_label]
    sampled_class = class_data.sample(n=min(n_samples, len(class_data)), random_state=SEED)
    sampled_data.append(sampled_class)

# Объединяем все отобранные данные
data_for_plot = pd.concat(sampled_data)

latent = data_for_plot[features_l].values
classes = data_for_plot["target"].values.astype(np.int8)

# t-SNE 2D
tsne = TSNE(n_components=2, random_state=SEED)
latent_2d = tsne.fit_transform(latent)
fig = plt.figure()
ax = fig.add_subplot()

# Разбиваем данные на отдельные компоненты
x = latent_2d[:, 0]
y = latent_2d[:, 1]

# Раскрашиваем точки в соответствии с классами
colors = ["b", "g", "y", "r"]  # Задаем цвета для каждого класса
class_names = ["No Event", "Operational Switching", "Abnormal Events", "Fault Events"]
for i in range(4):
    ax.scatter(
        x[classes == i],
        y[classes == i],
        c=colors[i],
        s=i + 1,
        label=class_names[i],
    )

ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
plt.title("2D Scatter Plot of t-SNE")
plt.legend()
plt.savefig("tsne_plot.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show(block=False)


# t-SNE 3D
# tsne = TSNE(n_components=3, random_state=SEED)
# latent_3d = tsne.fit_transform(latent)
# fig = plt.figure()
# ax = fig.add_subplot(
#     111,
#     projection="3d",
# )

# # Разбиваем данные на отдельные компоненты
# x = latent_3d[:, 0]
# y = latent_3d[:, 1]
# z = latent_3d[:, 2]

# # Раскрашиваем точки в соответствии с классами
# colors = ["b", "g", "y", "r"]  # Задаем цвета для каждого класса
# for i in range(4):
#     ax.scatter(
#         x[classes == i],
#         y[classes == i],
#         z[classes == i],
#         c=colors[i],
#         s=i + 1,
#         label=f"Class {i}",
#     )

# ax.set_xlabel("t-SNE Component 1")
# ax.set_ylabel("t-SNE Component 2")
# ax.set_zlabel("t-SNE Component 3")
# plt.title("3D Scatter Plot of t-SNE")
# plt.legend()
# plt.show(block=False)


# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score

# dbscan = DBSCAN(eps=1.2, min_samples=5)
# dbscan.fit(latent_3d)

# # -1 обозначает выбросы
# num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
# print("Количество кластеров, выделенных DBSCAN:", num_clusters)

# # Рассчитываем индекс силуэта для оценки качества кластеризации (необязательно)
# silhouette_avg = silhouette_score(latent_3d, dbscan.labels_)
# print("Средний индекс силуэта:", silhouette_avg)

# from scipy.stats import chi2_contingency

# # Подсчет таблицы сопряженности между классами DBSCAN и исходными классами
# contingency_table = pd.crosstab(dbscan.labels_, classes)

# # Рассчитываем коэффициент V Крамера
# chi2, p, dof, expected = chi2_contingency(contingency_table)
# n = contingency_table.sum().sum()
# phi2 = chi2 / n
# r, k = contingency_table.shape
# phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
# rcorr = r - ((r - 1) ** 2) / (n - 1)
# kcorr = k - ((k - 1) ** 2) / (n - 1)
# cramers_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# print("Коэффициент V Крамера:", cramers_v)

# # plot DBSCAN clusters
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # Разбиваем данные на отдельные компоненты
# x = latent_3d[:, 0]
# y = latent_3d[:, 1]
# z = latent_3d[:, 2]

# # Раскрашиваем точки в соответствии с кластерами, выделенными DBSCAN
# unique_labels = set(dbscan.labels_)
# for label in unique_labels:
#     if label == -1:
#         # Отображаем выбросы серым цветом
#         ax.scatter(
#             x[dbscan.labels_ == label],
#             y[dbscan.labels_ == label],
#             z[dbscan.labels_ == label],
#             c="gray",
#             s=20,
#             label="Outliers",
#         )
#     else:
#         # Отображаем точки каждого кластера с уникальным цветом
#         ax.scatter(
#             x[dbscan.labels_ == label],
#             y[dbscan.labels_ == label],
#             z[dbscan.labels_ == label],
#             label=f"Cluster {label}",
#             s=1,
#         )

# ax.set_xlabel("t-SNE Component 1")
# ax.set_ylabel("t-SNE Component 2")
# ax.set_zlabel("t-SNE Component 3")
# plt.title("3D Scatter Plot of DBSCAN Clusters")
# plt.legend()
# plt.show()


# # PCA 2d
# pca = PCA(n_components=2)
# latent_2d = pca.fit_transform(latent)

# # Создаем график
# fig = plt.figure()
# ax = fig.add_subplot()

# # Разбиваем данные на отдельные компоненты
# x = latent_3d[:, 0]
# y = latent_3d[:, 1]

# # Раскрашиваем точки в соответствии с классами
# colors = ["b", "g", "y", "r"]  # Задаем цвета для каждого класса
# for i in range(4):
#     ax.scatter(
#         x[classes == i],
#         y[classes == i],
#         c=colors[i],
#         s=i + 1,
#         label=f"Class {i}",
#     )

# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# plt.title("2D Scatter Plot of PCA")
# plt.show(block=False)


# # PCA 3d
# pca = PCA(n_components=3)
# latent_3d = pca.fit_transform(latent)

# # убираем лишние "хвосты"
# latent_3d = np.clip(latent_3d, -5, 5)

# # Создаем 3D-график
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # Разбиваем данные на отдельные компоненты
# x = latent_3d[:, 0]
# y = latent_3d[:, 1]
# z = latent_3d[:, 2]

# # Раскрашиваем точки в соответствии с классами
# colors = ["b", "g", "y", "r"]  # Задаем цвета для каждого класса
# for i in range(4):
#     ax.scatter(
#         x[classes == i],
#         y[classes == i],
#         z[classes == i],
#         c=colors[i],
#         s=i + 1,
#         label=f"Class {i}",
#     )

# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")
# plt.title("3D Scatter Plot of PCA")
# plt.show(block=False)


# correlation matrix
# one_hot_encoded = pd.get_dummies(data["target"], prefix="target").astype(int)
# data = pd.concat([data, one_hot_encoded], axis=1)
# matrix = data[
#     features_l
#     + [
#         "target_0",
#         "target_1",
#         "target_2",
#         "target_3",
#     ]
# ].values
# corr = np.corrcoef(matrix.T)

# plt.imshow(corr[BOTTLENECK_DIM:, :BOTTLENECK_DIM])
# plt.show(block=False)

pass
