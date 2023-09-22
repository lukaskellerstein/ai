import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import nltk
import re
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from torch.utils.data import Dataset

writer = SummaryWriter()

# -------------------
# Data
# -------------------
nltk.download("punkt")
nltk.download("stopwords")

# Load the data
test_data = pd.read_csv("data_run.csv")


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
test_data["text"] = test_data["text"].apply(preprocess_text)

# Generate Word2Vec model
word2vec = Word2Vec(test_data["text"], min_count=2, vector_size=300)


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


test_dataset = SentimentDataset(test_data, vectorize_text)
