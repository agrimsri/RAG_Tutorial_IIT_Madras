# A cross-encoder reads the query and a candidate chunk together.
# This is slower than vector search, but it can make a better relevance judgment
# because the model sees both texts at the same time.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _get_model():
    """
    Load the reranker only when it is first needed.

    Keeping this lazy makes normal indexing startup faster and lets students
    see reranking as a separate second-stage step in the RAG pipeline.
    """

    global _model

    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(RERANK_MODEL)

    return _model


def rerank(query, candidate_ids, chunk_lookup):
    """
    Rerank retrieved candidate chunks for a query.

    First-stage retrievers answer: "Which chunks might be relevant?"
    A reranker answers: "Of those candidates, which chunks best match
    this exact query?"
    """

    if not candidate_ids:
        return []

    pairs = [
        [query, chunk_lookup[chunk_id]]
        for chunk_id in candidate_ids
    ]

    scores = _get_model().predict(pairs)

    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        chunk_id
        for chunk_id, score in ranked
    ]
