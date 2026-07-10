from llm_client import embed


def blend_vectors(local_vector, document_vector, alpha=0.7):
    """
    Blend the chunk embedding with the document embedding.

    alpha = weight for chunk
    (1-alpha) = weight for document
    """

    blended = []

    for local, doc in zip(local_vector, document_vector):

        blended.append(alpha * local + (1 - alpha) * doc)

    return blended


def late_chunk_embeddings(document, chunks):

    # Whole document embedding
    document_vector = embed(document)

    vectors = []

    for chunk in chunks:

        chunk_vector = embed(chunk)

        vectors.append(
            blend_vectors(
                chunk_vector,
                document_vector
            )
        )

    return vectors