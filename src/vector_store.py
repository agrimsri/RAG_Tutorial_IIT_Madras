import os

import chromadb

from chunking import recursive_chunk
from late_chunking import late_chunk_embeddings

# Persist the collection on disk so embeddings survive restarts.
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db_milestone_3")
client = chromadb.PersistentClient(path=_DB_PATH)

collection = client.get_or_create_collection("mission_kb")


def is_indexed() -> bool:
    """Return True if the collection already contains embedded documents."""
    return collection.count() > 0


def index_documents(documents, ids):

    for document, doc_id in zip(documents, ids):

        chunks = recursive_chunk(document)

        vectors = late_chunk_embeddings(
            document,
            chunks
        )

        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):

            collection.add(
                ids=[f"{doc_id}_{i}"],
                embeddings=[vector],
                documents=[chunk]
            )


def retrieve(query, k=3):

    from llm_client import embed

    query_vector = embed(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    return results["documents"][0]