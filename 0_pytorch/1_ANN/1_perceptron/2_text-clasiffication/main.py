import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# -------------------------------------------------
# Perceptron
# for Classification
# -------------------------------------------------
# -------------------------------------------------

# Binary Classification = output is 0 or 1 (True or False, Yes or No, etc.)
# Binary sentiment classifier

# -------------------------------------------------
# Source: ChatGPT - GPT-4
# -------------------------------------------------

# Hyper-parameters
input_size = 1  # WILL BE REDEFINED BELOW IN THE CODE
hidden_size = 128
output_size = 1
num_epochs = 10
batch_size = 32
learning_rate = 0.001

training_eval_split_ratio = 0.7

# -------------------
# Data
# -------------------
sentences = [
    "I love the book",
    "this is an amazing book",
    "the fit of this dress is great",
    "I really like this series",
    "I feel very good about these beers",
    "this is my best work",
    "what an awesome view",
    "I do not like this dish",
    "I am tired of this stuff",
    "I can't deal with this",
    "he is my sworn enemy",
    "my boss is horrible",
    "I do not like the taste of this juice",
    "I can't stand the heat",
    "I've had a horrible day",
    "this is a horrible loss",
    "this game sucks",
    "I hate him",
    "I don't want to be here",
    "my life is hard",
    "I love the book",
]
labels = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
]  # 1 is positive, 0 is negative

# Build vocabulary
vocab = set()
for sentence in sentences:
    for word in sentence.split(" "):
        vocab.add(word)

vocab_size = len(vocab)

# Create mapping from words to indices
word_to_idx = {word: i for i, word in enumerate(vocab)}


# One-hot encoding of sentences
def encode_sentence(sentence):
    encoding = torch.zeros(vocab_size)
    for word in sentence.split(" "):
        encoding[word_to_idx[word]] = 1
    return encoding


sentences_encoded = torch.stack([encode_sentence(sentence) for sentence in sentences])
labels = torch.tensor(labels).reshape(-1, 1).float()

# Split the data into training and test sets
split_ratio = 0.7  # 70% for training
split = round(sentences_encoded.shape[0] * split_ratio)
X_train, X_test = sentences_encoded[:split], sentences_encoded[split:]
y_train, y_test = labels[:split], labels[split:]


# Prepare dataset
class MyDataset(Dataset):
    def __init__(self, sentence, sentiment):
        self.sentence = sentence
        self.sentiment = sentiment

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        return self.sentence[idx], self.sentiment[idx]


train_dataset = MyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = MyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------
# Model - Single layer Perceptron
# -------------------
class SLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, output_size), nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


input_size = vocab_size
model = SLP(input_size, output_size)


# -------------------------------------------------
# -------------------------------------------------
# EPOCHS - Training + Evaluation
# -------------------------------------------------
# -------------------------------------------------
# Loss function
criterion = nn.BCELoss()

# Finding optimal Loss function = Stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training function ------------
def train(dataloader):
    model.train()
    epoch_loss = 0

    n_total_steps = len(dataloader)

    for i, (sentence, sentiment) in enumerate(dataloader):
        # Forward pass
        outputs = model(sentence)
        loss = criterion(outputs, sentiment)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # print(f"Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    # return average loss for whole epoch
    return epoch_loss / len(dataloader)


# Evaluate function ------------
def evaluate(dataloader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (sentence, sentiment) in enumerate(dataloader):
            outputs = model(sentence)
            loss = criterion(outputs, sentiment)
            epoch_loss += loss.item()

    # return the average loss for whole epoch
    return epoch_loss / len(dataloader)


# -------------------
# EPOCHS CYCLE
# -------------------
train_losses = []
eval_losses = []
best_valid_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train(train_loader)
    valid_loss = evaluate(test_loader)

    # save losses
    train_losses.append(train_loss)
    eval_losses.append(valid_loss)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model, "saved_weights.pt")

    print("Epoch ", epoch + 1)
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\tVal Loss: {valid_loss:.5f}\n")

# -------------------
# Plotting results
# -------------------
# Plot of the training losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training loss")
plt.title("Training Losses")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_losses.png")

# Plot of the evaluation accuracies
plt.figure(figsize=(10, 5))
plt.plot(eval_losses, label="Test accuracy")
plt.title("Test Accuracies")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("evaluate_losses.png")


# -------------------
# Training
# -------------------
input_dim = vocab_size
lr = 0.01
epochs = 50


# criterion = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# losses = []
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()

#     y_pred = model(X_train)

#     loss = criterion(y_pred, y_train)
#     losses.append(loss.item())
#     loss.backward()

#     optimizer.step()

#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# # plot training loss
# plt.figure()
# plt.plot(range(epochs), losses)
# plt.xlabel("Epoch")
# plt.ylabel("Training Loss")
# plt.title("Training Loss over time")
# plt.savefig("training_loss.png")

# # -------------------
# # Evaluation
# # -------------------
# model.eval()
# y_pred_test = model(X_test)
# loss_test = criterion(y_pred_test, y_test)
# print(f"Test Loss: {loss_test.item():.4f}")

# # Binary accuracy
# y_pred_test = (y_pred_test > 0.5).float()
# accuracy = (y_pred_test == y_test).float().mean()
# print(f"Test Accuracy: {accuracy:.4f}")

# # plot histogram of predictions
# plt.figure()
# plt.hist(y_pred_test.detach().numpy(), bins=20)
# plt.xlabel("Prediction")
# plt.ylabel("Count")
# plt.title("Histogram of Model Predictions")
# plt.savefig("histogram_predictions.png")
