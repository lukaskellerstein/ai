import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset

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
