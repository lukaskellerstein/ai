from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# Insert one document
doc = {
    "title": "Python and Elasticsearch",
    "tags": ["python", "example"],
    "content": "Elasticsearch is great for full-text search."
}

es.index(index="my-index", id=1, document=doc)
