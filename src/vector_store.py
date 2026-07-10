import chromadb

from chunking import recursive_chunk
from contextual_retrieval import contextualize_chunk
from llm_client import embed

client = chromadb.Client()

collection = client.create_collection("mission_kb")


def index_documents(documents, ids):

    for document, doc_id in zip(documents, ids):

        chunks = recursive_chunk(document)

        for i, chunk in enumerate(chunks):

            contextual_chunk = contextualize_chunk(
                document,
                chunk
            )

            vector = embed(contextual_chunk)

            collection.add(
                ids=[f"{doc_id}_{i}"],
                embeddings=[vector],
                documents=[contextual_chunk]
            )


def retrieve(query, k=3):

    query_vector = embed(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    return results["documents"][0]