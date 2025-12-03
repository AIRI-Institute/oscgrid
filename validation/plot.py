import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from train import (
    BOTTLENECK_DIM,
)


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Liberation Sans']


SEED = 42

data = pd.read_csv("full_latents.csv")

data = data.dropna()
features_l = ["l" + str(i) for i in range(BOTTLENECK_DIM)]

# ===== SETTINGS =====
class_multipliers = {
    0: 2,  # For class 0: 2 times more than min_class_count
    1: 1,   # For class 1: standard amount
    2: 1,   # For class 2: standard amount
    3: 1,   # For class 3: standard amount
}
# =====================

# Find the minimum size among all classes (or only specified ones)
valid_classes = [cls for cls in class_multipliers if cls in data["target"].unique()]
if not valid_classes:
    raise ValueError("No specified classes found in data!")

min_class_count = min(
    len(data[data["target"] == cls]) // class_multipliers[cls] 
    for cls in valid_classes
)

# Collect data according to multipliers
sampled_data = []
for class_label, multiplier in class_multipliers.items():
    if class_label not in data["target"].unique():
        continue  # Skip missing classes
    
    n_samples = min_class_count * multiplier
    class_data = data[data["target"] == class_label]
    sampled_class = class_data.sample(n=min(n_samples, len(class_data)), random_state=SEED)
    sampled_data.append(sampled_class)

# Combine all sampled data
data_for_plot = pd.concat(sampled_data)

latent = data_for_plot[features_l].values
classes = data_for_plot["target"].values.astype(np.int8)

# t-SNE 2D
tsne = TSNE(n_components=2, random_state=SEED)
latent_2d = tsne.fit_transform(latent)
fig = plt.figure()
ax = fig.add_subplot()

# Split data into separate components
x = latent_2d[:, 0]
y = latent_2d[:, 1]

# Color points according to classes
colors = ["b", "g", "y", "r"]  # Set colors for each class
class_names = ["No Event", "Operational Switching", "Abnormal Events", "Fault Events"]
for i in range(4):
    ax.scatter(
        x[classes == i],
        y[classes == i],
        c=colors[i],
        s=5,
        label=class_names[i],
    )

ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
plt.title("2D Scatter Plot of t-SNE")
plt.legend()
plt.savefig("tsne_plot.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show(block=False)