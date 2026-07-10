from llm_client import generate


def build_cited_context(retrieved_chunks):
    """
    Convert retrieved chunks into numbered context blocks.

    The citation number is what the LLM will use in the final answer, for
    example: "Perseverance caches samples for return to Earth [1]."
    """

    context_blocks = []

    for i, item in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[{i}] Source: {item['source']}\n"
            f"{item['text']}"
        )

    return "\n\n".join(context_blocks)


def answer_with_citations(query, retrieved_chunks):
    """
    Generate an answer that is grounded in retrieved context.

    Teaching note:
    This is where RAG becomes more trustworthy. The prompt tells the model:
    - use only retrieved context,
    - cite the chunks it used,
    - say when the context is not enough.
    """

    context = build_cited_context(retrieved_chunks)

    prompt = f"""
You are answering questions using only the provided context.

Rules:
1. Use only facts from the context.
2. Cite every factual claim with the source number, like [1] or [2].
3. If the context does not contain the answer, say: "I do not know based on the provided context."
4. Do not mention sources that you did not use.

Question:
{query}

Context:
{context}

Answer:
"""

    return generate(prompt)
