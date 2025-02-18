import os
import chromadb

# -----------------------------------------
# Query ChromaDB
# -----------------------------------------

# Initialize ChromaDB client 
client = chromadb.PersistentClient(path="./chromadb") 

# Create a collection
my_collection = collection = client.get_collection("my_collection")

# query
results = collection.query(query_texts=["Jaké strategie jsou porovnávány mezi sebou?"], n_results=2)

print(results)


