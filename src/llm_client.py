import logging
import os
import time
from logging.handlers import RotatingFileHandler

import ollama

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.2")

# ── Logger setup ────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "llm_calls.log")

_logger = logging.getLogger("ollama_calls")
_logger.setLevel(logging.DEBUG)

if not _logger.handlers:
    _handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    )
    _logger.addHandler(_handler)
# ────────────────────────────────────────────────────────────────────────────


def embed(text: str):
    """
    Convert text into an embedding vector.
    """

    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )

    return response["embedding"]


def generate(prompt: str):
    """
    Generate text using the LLM.
    """
    _logger.info("GENERATE called | model=%s | prompt_len=%d | prompt_preview=%r",
                 GEN_MODEL, len(prompt), prompt[:200])

    t0 = time.perf_counter()
    try:
        response = ollama.chat(
            model=GEN_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _logger.error("GENERATE failed | model=%s | elapsed=%.3fs | error=%s",
                      GEN_MODEL, elapsed, exc)
        raise

    elapsed = time.perf_counter() - t0
    content = response["message"]["content"]
    _logger.info("GENERATE done   | model=%s | elapsed=%.3fs | response_len=%d | response_preview=%r",
                 GEN_MODEL, elapsed, len(content), content[:200])

    return content