import chromadb

from llm_client import embed
from chunking import recursive_chunk

client = chromadb.Client()

collection = client.create_collection("mission_kb")


def index_documents(documents, ids):

    for document, doc_id in zip(documents, ids):

        chunks = recursive_chunk(document)

        for i, chunk in enumerate(chunks):

            vector = embed(chunk)

            collection.add(
                ids=[f"{doc_id}_{i}"],
                embeddings=[vector],
                documents=[chunk]
            )


def retrieve(query, k=3):

    query_vector = embed(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    return results["documents"][0]