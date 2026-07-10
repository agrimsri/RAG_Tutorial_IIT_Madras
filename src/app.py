from document_loader import load_documents
from grounded_generation import answer_with_citations

from vector_store import (
    index_documents,
    is_indexed,
    retrieve_with_citations
)


# Laoding the files
documents, ids = load_documents()


# Only embed & index if not already cached on disk
if is_indexed():
    print("Knowledge Base already indexed — loading from cache.\n")
else:
    index_documents(documents, ids)
    print("Knowledge Base Indexed\n")

while True:

    query = input("Ask a question : ")

    # RAG retrieval now uses two stages:
    # 1. Hybrid search retrieves a larger candidate pool.
    # 2. A cross-encoder reranker reorders those candidates for the query.
    cited_results = retrieve_with_citations(query)

    # Grounded generation adds the final RAG step:
    # 3. Give the retrieved chunks to the LLM and require citations.
    answer = answer_with_citations(query, cited_results)

    print("\nGrounded Answer\n")

    print(answer)

    print("\nRetrieved Documents\n")

    for i, item in enumerate(cited_results, 1):

        print(f"\n----- Document {i}: {item['source']} -----\n")

        print(item["text"])
