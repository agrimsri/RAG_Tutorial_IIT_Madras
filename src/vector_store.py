import chromadb

from llm_client import embed

client = chromadb.Client()

collection = client.create_collection("mission_kb")


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