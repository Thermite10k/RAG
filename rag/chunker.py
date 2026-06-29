"""Stage 2 - Chunk.

In v1 the rule is simply: one slide = one chunk. Slides are already
semantically bounded units, and keeping chunk boundaries aligned to slides is
what makes neighbor expansion trivial (adjacency is just slide_num +/- 1 within
the same doc_id).

This stage is almost a pass-through today, but it stays a separate module so
that sub-slide chunking (splitting unusually text-dense slides) can be added
later without touching parsing or indexing.

Chunk schema
------------
{
    "chunk_id":    str,   # f"{doc_id}::{slide_num}", unique and stable
    "doc_id":      str,
    "slide_num":   int,
    "text":        str,
    "source_path": str,
}
"""

from __future__ import annotations


def make_chunk_id(doc_id: str, slide_num: int) -> str:
    return f"{doc_id}::{slide_num}"


def chunk_records(records: list[dict], drop_empty: bool = True) -> list[dict]:
    """Convert per-slide records into retrievable chunks.

    Parameters
    ----------
    records:
        Output of the parse stage.
    drop_empty:
        Skip slides whose text is empty (typically image-only slides). They
        carry no signal for embedding-based retrieval in v1. When OCR is added
        later, these will no longer be empty and will flow through normally.
    """
    chunks: list[dict] = []
    for r in records:
        text = (r.get("text") or "").strip()
        if drop_empty and not text:
            continue
        chunks.append(
            {
                "chunk_id": make_chunk_id(r["doc_id"], r["slide_num"]),
                "doc_id": r["doc_id"],
                "slide_num": r["slide_num"],
                "text": text,
                "source_path": r["source_path"],
            }
        )
    return chunks
