import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
import nltk
import re
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from torch.utils.data import Dataset

# -------------------
# Data
# -------------------
nltk.download("punkt")
nltk.download("stopwords")

# Load the data
data = pd.read_csv("data_train.csv")


# Preprocessing
def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Lowercase and split
    text = text.lower().split()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if word not in stop_words]

    return text


# Apply the preprocessing
data["text"] = data["text"].apply(preprocess_text)

# Split the data into train and test sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


# Generate Word2Vec model
word2vec = Word2Vec(data["text"], min_count=2, vector_size=300)


def vectorize_text(text):
    vectors = [word2vec.wv[word] for word in text if word in word2vec.wv.key_to_index]

    if vectors:  # If there are vectors
        return torch.FloatTensor(vectors).mean(dim=0)
    else:  # If there are no vectors, return a zero vector
        return torch.zeros(word2vec.vector_size)


class SentimentDataset(Dataset):
    def __init__(self, dataframe, vectorizer):
        self.dataframe = dataframe
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        text_vector = self.vectorizer(row["text"])
        sentiment = torch.FloatTensor(
            [row["sentiment"]]
        )  # assuming 'sentiment' column is numerical: 0 (negative) or 1 (positive)

        # print("----------------")
        # print("text_vector")
        # print(text_vector.shape)
        # # print(text_vector)
        # print("sentiment")
        # print(sentiment.shape)
        # # print(sentiment)

        return text_vector, sentiment


train_dataset = SentimentDataset(train_data, vectorize_text)
valid_dataset = SentimentDataset(val_data, vectorize_text)
