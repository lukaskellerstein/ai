import torch
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from plots import plot_time_series
from helpers import denormalize_data

writer = SummaryWriter()

seq_len = 20

# -------------------
# -------------------
# 1. Get data
# -------------------
# -------------------

df = pd.read_csv("TSLA.csv", index_col=0)

# You can also use all input data (OHLC Volume)
# to predict the next close price
# -> OHLC = 4 input features
# -> C = 1 input feature

# remove unecessary columns
df.drop("open", axis=1, inplace=True)
df.drop("high", axis=1, inplace=True)
df.drop("low", axis=1, inplace=True)
df.drop("volume", axis=1, inplace=True)
df.drop("average", axis=1, inplace=True)
df.drop("barCount", axis=1, inplace=True)

print("df")
print(df)

# Plotting data
plot_data = {"real_price": df}
figure = plot_time_series(plot_data, "Origin price", "Date", "Real Price")
writer.add_figure("DATA/Price", figure)

# -------------------
# Normalize data
# -------------------

# create a new dataframe
df_normalized = df.copy()

min_max_scaler = MinMaxScaler()
# df_normalized["open"] = min_max_scaler.fit_transform(df.open.values.reshape(-1, 1))
# df_normalized["high"] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
# df_normalized["low"] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
df_normalized["close"] = min_max_scaler.fit_transform(df.close.values.reshape(-1, 1))


print("df_normalized")
print(df_normalized)

# Plotting data
plot_data = {"norm_price": df_normalized}
figure = plot_time_series(plot_data, "Origin price", "Date", "Normalized Price")
writer.add_figure("DATA/Norm", figure)


# -------------------
# Sequence data
# -------------------

data = df_normalized[["date", "close"]].values
print("data")
print(data)

sequences = []
for index in range(len(data) - seq_len):
    sequences.append(data[index : index + seq_len])
sequences = np.array(sequences)

print("sequences")
print(len(sequences))

# -------------------
# Splitting data
# -------------------
valid_set_size_percentage = 10
test_set_size_percentage = 10

valid_set_size = int(np.round(valid_set_size_percentage / 100 * sequences.shape[0]))
test_set_size = int(np.round(test_set_size_percentage / 100 * sequences.shape[0]))
train_set_size = sequences.shape[0] - (valid_set_size + test_set_size)

x_train = sequences[:train_set_size, :-1, :]
y_train = sequences[:train_set_size, -1, :]
x_valid = sequences[train_set_size : train_set_size + valid_set_size, :-1, :]
y_valid = sequences[train_set_size : train_set_size + valid_set_size, -1, :]
x_test = sequences[train_set_size + valid_set_size :, :-1, :]
y_test = sequences[train_set_size + valid_set_size :, -1, :]


# for i in range(1):
#     print("x_train")
#     print(x_train[i])
#     print("y_train")
#     print(y_train[i])


# Plotting data
df_y_train = pd.DataFrame(y_train, columns=["date", "close"])
df_y_valid = pd.DataFrame(y_valid, columns=["date", "close"])
df_y_test = pd.DataFrame(y_test, columns=["date", "close"])

# plot
plot_data = {
    "train": df_y_train,
    "valid": df_y_valid,
    "test": df_y_test,
}
figure = plot_time_series(plot_data, "Phases", "Date", "Normalized Price")
writer.add_figure("DATA-SPLIT/Price", figure)

df_y_train_norm = denormalize_data(df_y_train, min_max_scaler)
df_y_valid_norm = denormalize_data(df_y_valid, min_max_scaler)
df_y_test_norm = denormalize_data(df_y_test, min_max_scaler)

plot_data = {
    "train": df_y_train_norm,
    "valid": df_y_valid_norm,
    "test": df_y_test_norm,
}
figure = plot_time_series(plot_data, "Phases", "Date", "Real Price")
writer.add_figure("DATA-SPLIT/Norm", figure)


# -------------------
# Creating data loaders
# -------------------
x_train_final = np.array(x_train[:, :, -1]).astype(np.float32)
y_train_final = np.array(y_train[:, -1]).astype(np.float32)
x_valid_final = np.array(x_valid[:, :, -1]).astype(np.float32)
y_valid_final = np.array(y_valid[:, -1]).astype(np.float32)
x_test_final = np.array(x_test[:, :, -1]).astype(np.float32)
y_test_final = np.array(y_test[:, -1]).astype(np.float32)


train_dataset = TensorDataset(torch.tensor(x_train_final), torch.tensor(y_train_final))
valid_dataset = TensorDataset(torch.tensor(x_valid_final), torch.tensor(y_valid_final))
test_dataset = TensorDataset(torch.tensor(x_test_final), torch.tensor(y_test_final))
