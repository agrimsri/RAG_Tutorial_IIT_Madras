from document_loader import load_documents

from vector_store import (
    index_documents,
    retrieve
)


# Laoding the files
documents, ids = load_documents()


#Indexing the documents
index_documents(documents, ids)

print("Knowledge Base Indexed\n")

while True:

    query = input("Ask a question : ")

    results = retrieve(query)

    print("\nRetrieved Documents\n")

    for i, doc in enumerate(results, 1):

        print(f"\n----- Document {i} -----\n")

        print(doc)