from document_loader import load_documents

from vector_store import (
    index_documents,
    is_indexed,
    hybrid_retrieve
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

    results = hybrid_retrieve(query)

    print("\nRetrieved Documents\n")

    for i, doc in enumerate(results, 1):

        print(f"\n----- Document {i} -----\n")

        print(doc)