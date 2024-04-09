import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType


# -------------------------------------------
# Connect to local Weaviate
# -------------------------------------------

client = weaviate.connect_to_local()

# -------------------------------------------
# Create table/collection
# -------------------------------------------
client.collections.create("Books", 
                          vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
                          properties=[
                                Property(name="title", data_type=DataType.TEXT),
                                Property(name="body", data_type=DataType.TEXT),
                            ])

client.close()