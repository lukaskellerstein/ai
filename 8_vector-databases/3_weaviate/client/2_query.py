import weaviate
import weaviate.classes as wvc


# -------------------------------------------
# Connect to local Weaviate
# -------------------------------------------

client = weaviate.connect_to_local()

try:
    db_books = client.collections.get("Books")


    # -------------------------------------------
    # Basic search
    # -------------------------------------------
    # response = db_books.query.fetch_objects()

    # for o in response.objects:
    #     print(o.uuid)
    #     print(o.properties)

    # -------------------------------------------
    # Vector search = similarity search
    # -------------------------------------------
    response = db_books.query.near_text(
        query="animals in movies",
        limit=2,
        return_metadata=wvc.query.MetadataQuery(distance=True)
    )

    for o in response.objects:
        print(o.uuid)
        print(o.properties)
        print(o.metadata.distance)


finally:
    client.close()