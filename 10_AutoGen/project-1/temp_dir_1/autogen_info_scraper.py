# filename: autogen_info_scraper.py

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# Download the NLTK English stopwords if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# URL of the AutoGen project
url = "https://www.gnu.org/software/autogen/"

# Send HTTP request to the URL
response = requests.get(url)

# Parse the HTML content of the page with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the text from the HTML content
text = soup.get_text()

# Remove any non-alphanumeric characters
text = re.sub(r'\W', ' ', text)

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Tokenize the text into words
words = word_tokenize(text.lower())

# Remove stopwords from the list of words
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

# Create a frequency table for the words
freq_table = dict()
for word in words:
    if word in freq_table:
        freq_table[word] += 1
    else:
        freq_table[word] = 1

# Score the sentences based on the frequency table
sentence_scores = dict()
for sentence in sentences:
    for word, freq in freq_table.items():
        if word in sentence.lower():
            if sentence in sentence_scores:
                sentence_scores[sentence] += freq
            else:
                sentence_scores[sentence] = freq

# Get the 7 highest scoring sentences
summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:7]

# Join the sentences together to form the summary
summary = ' '.join(summary_sentences)

# Write the summary to a markdown file
with open('autogen-summary.md', 'w') as f:
    f.write(summary)