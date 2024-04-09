import weaviate
import os

# -------------------------------------------
# Connect to local Weaviate
# -------------------------------------------

client = weaviate.connect_to_local()

# -------------------------------------------
# Read documents from source directory
# Split the document if it exceeds the maximum length
# Save each part into Weaviate
# -------------------------------------------
db_books = client.collections.get("Books")


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
            
            # Save each part into ChromaDB
            for index, part in enumerate(parts):
                data = {
                    "title": filename,
                    "body": part
                }
                uuid = db_books.data.insert(data)
                print(f"Document {filename} part {index} saved successfully with UUID: {uuid}")





client.close()  # Close client gracefully