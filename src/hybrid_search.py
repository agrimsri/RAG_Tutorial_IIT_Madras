from rank_bm25 import BM25Okapi


class HybridSearch:

    def __init__(self):

        self.chunk_ids = []

        self.chunk_texts = []

        self.bm25 = None

    def build_index(self, ids, chunks):

        self.chunk_ids = ids

        self.chunk_texts = chunks

        tokenized = [
            chunk.split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

    def bm25_search(self, query, k=5):

        scores = self.bm25.get_scores(
            query.split()
        )

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [
            self.chunk_ids[i]
            for i in ranked[:k]
        ]

    def reciprocal_rank_fusion(
        self,
        vector_ids,
        bm25_ids,
        k=60
    ):

        scores = {}

        for rank, doc in enumerate(vector_ids):

            scores[doc] = scores.get(doc, 0)

            scores[doc] += 1 / (k + rank + 1)

        for rank, doc in enumerate(bm25_ids):

            scores[doc] = scores.get(doc, 0)

            scores[doc] += 1 / (k + rank + 1)

        ranked = sorted(
            scores,
            key=scores.get,
            reverse=True
        )

        return ranked