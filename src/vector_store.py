import chromadb

from chunking import recursive_chunk
from late_chunking import late_chunk_embeddings

client = chromadb.Client()

collection = client.create_collection("mission_kb")


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