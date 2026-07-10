from llm_client import generate


def contextualize_chunk(document: str, chunk: str) -> str:
    """
    Ask the LLM to write 1–2 sentences explaining
    where this chunk belongs in the document.
    """

    prompt = f"""
You are improving a document chunk for retrieval.

Full Document:
{document}

Chunk:
{chunk}

Write 1-2 sentences that explain the context
of this chunk.

Mention important names like
missions, astronauts, agencies, etc.

Return ONLY the context.
"""

    context = generate(prompt).strip()

    return f"{context}\n\n{chunk}"