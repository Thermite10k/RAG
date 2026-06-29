"""Stage 4 - Query & retrieve.

Given a question:
  1. embed it with the same model the index was built with,
  2. score it against every chunk (cosine == dot product, vectors normalized),
  3. take the top-k *seed* chunks,
  4. expand each seed to its neighbors (same doc, slide_num +/- window),
  5. order the final set by document then slide number (study material reads
     best in its original order, not in relevance order),
  6. assemble a provenance-tagged context blob: the question followed by each
     chunk prefixed with its source.

The output is a plain text file that *is* the prompt. In a later version this
same stage hands the blob to a local or API LLM instead of writing it to disk -
a change confined to this module.

Note the k vs window split: k controls how many independent hits seed the
result; window controls how much surrounding context each seed drags in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .indexer import QUERY_INSTRUCTION, Index


@dataclass
class Retrieved:
    chunk: dict
    score: float       # similarity of this chunk's own seed (neighbors inherit the seed score)
    is_seed: bool      # True if retrieved directly, False if pulled in as a neighbor


def embed_query(
    query: str,
    model: SentenceTransformer,
    instruction: str = QUERY_INSTRUCTION,
) -> np.ndarray:
    """Embed a single query string into a normalized 1-D vector."""
    vec = model.encode(
        [instruction + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vec.astype(np.float32)


def retrieve(query_vec: np.ndarray, index: Index, k: int = 5) -> list[tuple[dict, float]]:
    """Return the top-k (chunk, score) pairs by cosine similarity."""
    sims = index.embeddings @ query_vec  # both sides are L2-normalized
    k = min(k, len(index.chunks))
    top_idx = np.argsort(-sims)[:k]
    return [(index.chunks[i], float(sims[i])) for i in top_idx]


def expand_neighbors(
    hits: list[tuple[dict, float]],
    index: Index,
    window: int = 1,
) -> list[Retrieved]:
    """Add the +/- window slide neighbors of each hit, deduped (seeds win)."""
    by_key = {(c["doc_id"], c["slide_num"]): c for c in index.chunks}
    selected: dict[str, Retrieved] = {}

    # Seeds first, so a chunk that is both a seed and someone's neighbor stays a seed.
    for chunk, score in hits:
        selected[chunk["chunk_id"]] = Retrieved(chunk=chunk, score=score, is_seed=True)

    for chunk, score in hits:
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            neighbor = by_key.get((chunk["doc_id"], chunk["slide_num"] + delta))
            if neighbor is None:
                continue
            if neighbor["chunk_id"] in selected:
                continue  # already present (as seed or another neighbor)
            selected[neighbor["chunk_id"]] = Retrieved(
                chunk=neighbor, score=score, is_seed=False
            )

    return list(selected.values())


def order_for_reading(items: list[Retrieved]) -> list[Retrieved]:
    """Order by document then slide number - the natural reading order."""
    return sorted(items, key=lambda r: (r.chunk["doc_id"], r.chunk["slide_num"]))


def assemble_context(question: str, items: list[Retrieved]) -> str:
    """Build the question + provenance-tagged context string."""
    ordered = order_for_reading(items)
    parts = [f"QUESTION:\n{question}\n", "CONTEXT:"]
    for r in ordered:
        tag = "seed" if r.is_seed else "neighbor"
        header = (
            f"[{r.chunk['doc_id']} - slide {r.chunk['slide_num']} "
            f"| {tag} | score {r.score:.3f}]"
        )
        parts.append(f"\n{header}\n{r.chunk['text']}")
    return "\n".join(parts) + "\n"


def answer_to_file(
    question: str,
    index: Index,
    model: SentenceTransformer,
    out_path: str | Path,
    k: int = 5,
    window: int = 1,
) -> Path:
    """End-to-end convenience: query -> retrieve -> expand -> write context.txt."""
    query_vec = embed_query(question, model)
    hits = retrieve(query_vec, index, k=k)
    expanded = expand_neighbors(hits, index, window=window)
    context = assemble_context(question, expanded)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(context, encoding="utf-8")
    return out_path
