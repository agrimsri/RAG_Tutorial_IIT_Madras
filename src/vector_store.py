import os

import chromadb

from chunking import recursive_chunk
from contextual_retrieval import contextualize_chunk
from llm_client import embed
from hybrid_search import HybridSearch

# Persist the collection on disk so embeddings survive restarts.
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db_milestone_5")
client = chromadb.PersistentClient(path=_DB_PATH)

collection = client.get_or_create_collection("mission_kb")

# Initialize hybrid search and chunk lookup
hybrid = HybridSearch()
chunk_lookup = {}

def is_indexed() -> bool:
    """Return True if the collection already contains embedded documents."""
    return collection.count() > 0

# If the database is already indexed, restore the local caches
if is_indexed():
    stored = collection.get()
    if stored and stored.get("ids"):
        for chunk_id, doc in zip(stored["ids"], stored["documents"]):
            chunk_lookup[chunk_id] = doc
        hybrid.build_index(stored["ids"], stored["documents"])



def index_documents(documents, ids):

    all_chunk_ids = []

    all_chunks = []

    for document, doc_id in zip(documents, ids):

        chunks = recursive_chunk(document)

        for i, chunk in enumerate(chunks):

            contextual_chunk = contextualize_chunk(
                document,
                chunk
            )

            chunk_id = f"{doc_id}_{i}"

            chunk_lookup[chunk_id] = contextual_chunk

            vector = embed(contextual_chunk)

            collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[contextual_chunk]
            )

            all_chunk_ids.append(chunk_id)

            all_chunks.append(contextual_chunk)

    hybrid.build_index(
        all_chunk_ids,
        all_chunks
    )


def hybrid_retrieve(query, k=5):

    query_vector = embed(query)

    vector_results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    vector_ids = vector_results["ids"][0]

    bm25_ids = hybrid.bm25_search(
        query,
        k
    )

    fused = hybrid.reciprocal_rank_fusion(
        vector_ids,
        bm25_ids
    )

    return [
        chunk_lookup[i]
        for i in fused[:k]
    ]