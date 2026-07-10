import os
import ollama

MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def embed(text: str):
    """
    Convert text into an embedding vector.
    """

    response = ollama.embeddings(
        model=MODEL,
        prompt=text
    )

    return response["embedding"]