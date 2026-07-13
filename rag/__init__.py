"""A simple RAG-over-PDF study tool, built as four discrete stages.

    parse  ->  chunk  ->  index  ->  query

Each stage reads the previous stage's artifact and writes its own, so OCR and a
real LLM answerer can be added later by editing a single stage.
"""

from .pdf_parser import (
    parse_pdf,
    parse_directory,
    save_records,
    load_records,
)
from .chunker import chunk_records, make_chunk_id
from .indexer import (
    Index,
    DEFAULT_MODEL,
    load_model,
    build_index,
    save_index,
    load_index,
)
from .query import (
    Retrieved,
    embed_query,
    retrieve,
    expand_neighbors,
    order_for_reading,
    assemble_context,
    answer_to_file,
)

from .llm import prompt_for_key, get_answer

__all__ = [
    "parse_pdf",
    "parse_directory",
    "save_records",
    "load_records",
    "chunk_records",
    "make_chunk_id",
    "Index",
    "DEFAULT_MODEL",
    "load_model",
    "build_index",
    "save_index",
    "load_index",
    "Retrieved",
    "embed_query",
    "retrieve",
    "expand_neighbors",
    "order_for_reading",
    "assemble_context",
    "answer_to_file",
    "prompt_for_key",
    "get_answer"
]
