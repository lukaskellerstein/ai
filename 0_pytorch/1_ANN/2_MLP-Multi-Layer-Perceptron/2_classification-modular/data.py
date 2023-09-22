import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from helpers import plot_to_image

writer = SummaryWriter()

# -------------------
# Data
# -------------------

# Load spambase dataset
data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
    header=None,
)

X = data.values[:, :-1]  # all columns except the last one
y = data.values[:, -1]  # only the last column

# Normalize data
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Split data into training, validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# Dataset
class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = SpamDataset(X_train, y_train)
valid_dataset = SpamDataset(X_valid, y_valid)
test_dataset = SpamDataset(X_test, y_test)


# -------------------
# Visualizations
# -------------------


# Converting the data back to DataFrame for convenience
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y

# 1. Histograms for each feature
figure = plt.figure(figsize=(20, 15))
df.hist(bins=50, ax=figure.gca())
plt.tight_layout()
image = plot_to_image(figure)
writer.add_image("Feature Histograms", image, 0)

# 2. Correlation heatmap
figure = plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), ax=figure.gca())
image = plot_to_image(figure)
writer.add_image("Correlation Heatmap", image, 0)

# 3. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

figure = plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
image = plot_to_image(figure)
writer.add_image("PCA Plot", image, 0)

# 4. t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

figure = plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
image = plot_to_image(figure)
writer.add_image("t-SNE Plot", image, 0)

# Close the writer
writer.close()
