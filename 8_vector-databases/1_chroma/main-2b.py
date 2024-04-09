import os
import chromadb

# -----------------------------------------
# Write into ChromaDB
# and save on disk
# -----------------------------------------


# Initialize ChromaDB client 
client = chromadb.PersistentClient(path="./chromadb") 

# Create a collection
my_collection = collection = client.get_collection("my_collection")

# query
results = collection.query(query_texts=["Who is forrest?"], n_results=2)

print(results)


