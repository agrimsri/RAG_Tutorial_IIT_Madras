import os

import chromadb

from chunking import recursive_chunk
from contextual_retrieval import contextualize_chunk
from llm_client import embed
from hybrid_search import HybridSearch
from reranker import rerank

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


def _source_from_chunk_id(chunk_id):
    """
    Convert a chunk id back into the original source document name.

    Example:
    "01_perseverance_3" -> "01_perseverance"
    """

    return chunk_id.rsplit("_", 1)[0]


def _retrieve_chunk_ids(query, k=5, candidate_k=20, use_reranking=True):
    """
    Return ranked chunk ids instead of chunk text.

    Keeping this separate lets us support both:
    - plain retrieval demos, which only print chunk text,
    - grounded generation, which needs chunk ids for citations.
    """

    query_vector = embed(query)

    vector_results = collection.query(
        query_embeddings=[query_vector],
        n_results=candidate_k
    )

    vector_ids = vector_results["ids"][0]

    bm25_ids = hybrid.bm25_search(
        query,
        candidate_k
    )

    fused = hybrid.reciprocal_rank_fusion(
        vector_ids,
        bm25_ids
    )

    candidate_ids = fused[:candidate_k]

    if use_reranking:
        final_ids = rerank(
            query,
            candidate_ids,
            chunk_lookup
        )
    else:
        final_ids = candidate_ids

    return final_ids[:k]



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


def hybrid_retrieve(query, k=5, candidate_k=20, use_reranking=True):
    """
    Retrieve chunks for RAG.

    Teaching note:
    - k is the number of chunks we finally give to the LLM.
    - candidate_k is the larger pool we retrieve cheaply before reranking.
    - reranking is a second-stage filter that spends more compute only on the
      most promising candidates.
    """

    final_ids = _retrieve_chunk_ids(
        query,
        k,
        candidate_k,
        use_reranking
    )

    return [
        chunk_lookup[i]
        for i in final_ids
    ]


def retrieve_with_citations(query, k=5, candidate_k=20, use_reranking=True):
    """
    Retrieve chunks and keep citation metadata.

    Each returned item has:
    - id: exact chunk id
    - source: original document name
    - text: retrieved chunk text
    """

    final_ids = _retrieve_chunk_ids(
        query,
        k,
        candidate_k,
        use_reranking
    )

    return [
        {
            "id": chunk_id,
            "source": _source_from_chunk_id(chunk_id),
            "text": chunk_lookup[chunk_id]
        }
        for chunk_id in final_ids
    ]
