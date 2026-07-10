from pathlib import Path


def load_documents(folder="Data"):

    documents = []
    ids = []

    folder = Path(folder)

    for file in folder.glob("*.txt"):

        documents.append(file.read_text())

        ids.append(file.stem)

    return documents, ids