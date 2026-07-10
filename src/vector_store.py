import os

import chromadb

from llm_client import embed

# Persist the collection on disk so embeddings survive restarts.
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db_milestone_1")
client = chromadb.PersistentClient(path=_DB_PATH)

collection = client.get_or_create_collection("mission_kb")


def is_indexed() -> bool:
    """Return True if the collection already contains embedded documents."""
    return collection.count() > 0


def index_documents(documents, ids):

    for doc, doc_id in zip(documents, ids):

        vector = embed(doc)

        collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[doc]
        )


def retrieve(query, k=3):

    query_vector = embed(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    return results["documents"][0]