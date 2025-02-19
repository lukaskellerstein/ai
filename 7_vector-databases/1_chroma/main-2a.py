import os
import chromadb
from chromadb.utils import embedding_functions
import uuid

# -----------------------------------------
# Write into ChromaDB
# and save on disk
# -----------------------------------------

# Initialize ChromaDB client 
client = chromadb.PersistentClient(path="chromadb") 

# embeding
embedding = embedding_functions.DefaultEmbeddingFunction()

# Create a collection
my_collection = client.get_or_create_collection( 
    name="my_collection", 
    embedding_function=embedding, 
) 

def split_document(text, max_length=2048):
    """Split the document into parts with a maximum length."""
    words = text.split()
    parts = []
    current_part = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > max_length:
            parts.append(' '.join(current_part))
            current_part = [word]
            current_length = len(word)
        else:
            current_part.append(word)
            current_length += len(word)
    parts.append(' '.join(current_part))  # Add the last part
    return parts


# Path to the directory containing the .txt files
source_dir = 'source'

for filename in os.listdir(source_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(source_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Check if the document needs to be split
            if len(content) > 2048:
                parts = split_document(content)
            else:
                parts = [content]
            
            ids = [str(uuid.uuid4()) for _ in parts]

            # Save each part into ChromaDB
            for part in parts:
                my_collection.add(documents=parts, ids=ids)


