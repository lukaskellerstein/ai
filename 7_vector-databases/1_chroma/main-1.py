import chromadb

chroma_client = chromadb.Client()

# ---------------------
# Collections
# ---------------------

# create
collection = chroma_client.create_collection(name="my_collection")

# get
# collection = client.get_collection(name="test") # Get a collection object from an existing collection, by name. Will raise an exception if it's not found.

# get or create
# collection = client.get_or_create_collection(name="test") # Get a collection object from an existing collection, by name. If it doesn't exist, create it.

# delete
# client.delete_collection(name="my_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible


# ---------------------
# Documents
# ---------------------

# Add documents to the collection
collection.add(
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"],
)

# Update documents in the collection
collection.update(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[
        {"chapter": "3", "verse": "16"},
        {"chapter": "3", "verse": "5"},
        {"chapter": "29", "verse": "11"},
        ...,
    ],
    documents=["doc1", "doc2", "doc3", ...],
)

# Upsert documents in the collection
collection.upsert(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[
        {"chapter": "3", "verse": "16"},
        {"chapter": "3", "verse": "5"},
        {"chapter": "29", "verse": "11"},
        ...,
    ],
    documents=["doc1", "doc2", "doc3", ...],
)

# Delete documents from the collection
collection.delete(ids=["id1", "id2", "id3", ...], where={"chapter": "20"})


# Query the collection
results = collection.query(query_texts=["This is a query"], n_results=2)

print(results)


# ---------------------
# Existing Images from disk (or blob storage)
# ---------------------
# ???


# ---------------------
# Existing Documents from disk (or blob storage)
# ---------------------
# ???
