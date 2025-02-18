from typing import Annotated
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIEmbeddingPromptExecutionSettings,
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.memory.in_memory import InMemoryVectorCollection
from semantic_kernel.data import (
    vectorstoremodel, 
    VectorStoreRecordVectorField, 
    IndexKind, 
    DistanceFunction,
    VectorStoreRecordKeyField,
    VectorStoreRecordDataField,
    VectorStoreRecordUtils,
    VectorSearchOptions,
    VectorSearchFilter
)
from dataclasses import dataclass, field
import numpy as np
from uuid import uuid4
import asyncio

# Data model
@vectorstoremodel
@dataclass
class DataModelArray:
    vector: Annotated[
        np.ndarray | None,
        VectorStoreRecordVectorField(
            embedding_settings={"emb": OpenAIEmbeddingPromptExecutionSettings(dimensions=1536)},
            index_kind=IndexKind.HNSW,
            dimensions=1536,
            distance_function=DistanceFunction.COSINE_SIMILARITY,
            property_type="float",
            serialize_function=np.ndarray.tolist,
            deserialize_function=np.array,
        ),
    ] = None
    id: Annotated[str, VectorStoreRecordKeyField()] = field(default_factory=lambda: str(uuid4()))
    content: Annotated[
        str,
        VectorStoreRecordDataField(
            has_embedding=True,
            embedding_property_name="vector",
            property_type="str",
            is_full_text_searchable=True,
        ),
    ] = "content1"
    title: Annotated[str, VectorStoreRecordDataField(property_type="str", is_full_text_searchable=True)] = "title"
    tag: Annotated[str, VectorStoreRecordDataField(property_type="str", is_filterable=True)] = "tag"


async def main():
    print("-" * 30)
    kernel = Kernel()
    embedder = OpenAITextEmbedding(service_id="emb", ai_model_id="text-embedding-3-small")
    kernel.add_service(embedder)

    db = InMemoryVectorCollection("testcollection", DataModelArray)

    # create collection if not exists
    await db.create_collection_if_not_exists()

    # Add records to DB
    record1 = DataModelArray(
        content="Semantic Kernel is awesome",
        id="e6103c03-487f-4d7d-9c23-4723651c17f4",
        title="Overview",
        tag="general",
    )
    record2 = DataModelArray(
        content="Semantic Kernel is available in dotnet, python and Java.",
        id="09caec77-f7e1-466a-bcec-f1d51c5b15be",
        title="Semantic Kernel Languages",
        tag="general",
    )

    records = await VectorStoreRecordUtils(kernel).add_vector_to_records(
        [record1, record2], data_model_type=DataModelArray
    )
    keys = await db.upsert_batch(records)
    print(f"    Upserted {keys=}")

    # Get Records from DB
    print("Getting records!")
    results = await db.get_batch([record1.id, record2.id])
    if results:
        [print(result) for result in results]
    else:
        print("Nothing found...")


    # Search records
    options = VectorSearchOptions(
        vector_field_name="vector",
        include_vectors=True,
        filter=VectorSearchFilter.equal_to("tag", "general"),
    )

    # TEXT SEARCH
    print("-" * 30)
    print("Using text search")
    try:
        search_results = await db.text_search("python", options)
        if search_results.total_count == 0:
            print("\nNothing found...\n")
        else:
            [print(result) async for result in search_results.results]
    except Exception:
        print("Text search could not execute.")


    # VECTOR SEARCH
    print("-" * 30)
    print(
        "Using vectorized search, depending on the distance function, "
        "the better score might be higher or lower."
    )
    try:
        search_results = await db.vectorized_search(
            vector=(await embedder.generate_raw_embeddings(["python"]))[0],
            options=VectorSearchOptions(vector_field_name="vector", include_vectors=True),
        )
        if search_results.total_count == 0:
            print("\nNothing found...\n")
        else:
            [print(result) async for result in search_results.results]
    except Exception:
        print("Vectorized search could not execute.")


    # DELETE COLLECTION
    # print("-" * 30)
    # print("Deleting collection!")
    # await db.delete_collection()


if __name__ == "__main__":
    asyncio.run(main())



