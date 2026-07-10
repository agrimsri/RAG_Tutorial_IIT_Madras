from typing import List


def fixed_chunk(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def recursive_chunk(text: str, max_chunk_size: int = 300) -> List[str]:
    """
    A simple recursive chunker.

    Strategy:
    1. Split by paragraphs.
    2. If paragraph is too large,
       split it using fixed chunking.
    """

    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:

        if len(paragraph) <= max_chunk_size:

            chunks.append(paragraph)

        else:

            chunks.extend(
                fixed_chunk(
                    paragraph,
                    chunk_size=max_chunk_size,
                    overlap=50
                )
            )

    return chunks