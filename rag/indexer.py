"""Stage 3 - Embed & index.

Embed each chunk's text into a vector with a local sentence-transformer model
(no API key, runs on CPU). Embeddings are L2-normalized so that cosine
similarity is just a dot product at query time.

For a single course (hundreds to low-thousands of chunks) a plain NumPy matrix
plus the chunk list is more than enough and trivial to debug - no FAISS / Chroma
needed yet. The persisted artifact is two files:

    <prefix>.npy        float32 matrix, shape (n_chunks, dim), row i <-> chunks[i]
    <prefix>.chunks.json   the chunk list + the model name used

Keeping the model name in the artifact matters: a query MUST be embedded with
the same model the index was built with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# bge-small is a strong, small, CPU-friendly retrieval model. bge models expect
# a short instruction prepended to the *query* (not the passages) for retrieval.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass
class Index:
    """An in-memory index: the embedding matrix, its chunks, and the model name."""

    embeddings: np.ndarray  # (n_chunks, dim), float32, L2-normalized
    chunks: list[dict]
    model_name: str

    def __len__(self) -> int:
        return len(self.chunks)


def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load (and on first run, download) the embedding model."""
    return SentenceTransformer(model_name)


def build_index(
    chunks: list[dict],
    model: SentenceTransformer,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Index:
    """Embed all chunk texts and return an Index."""
    if not chunks:
        raise ValueError("No chunks to index. Did the parse/chunk stage drop everything?")

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    ).astype(np.float32)

    return Index(embeddings=embeddings, chunks=chunks, model_name=model_name)


def save_index(index: Index, prefix: str | Path) -> tuple[Path, Path]:
    """Persist the index to <prefix>.npy and <prefix>.chunks.json."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    emb_path = prefix.with_suffix(".npy")
    meta_path = prefix.with_suffix(".chunks.json")

    np.save(emb_path, index.embeddings)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"model_name": index.model_name, "chunks": index.chunks},
            f,
            ensure_ascii=False,
        )
    return emb_path, meta_path


def load_index(prefix: str | Path) -> Index:
    """Load an index previously written with save_index."""
    prefix = Path(prefix)
    embeddings = np.load(prefix.with_suffix(".npy"))
    with prefix.with_suffix(".chunks.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return Index(
        embeddings=embeddings.astype(np.float32),
        chunks=meta["chunks"],
        model_name=meta["model_name"],
    )
