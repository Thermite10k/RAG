"""Stage 1 - Parse.

Turn .pdf files into structured per-slide records. Each record is one slide /
page and carries the metadata everything downstream depends on (slide number,
source path) plus the extracted text and any PDF annotation/comment text.

The output artifact is JSONL: one JSON object per line. JSONL is chosen over
Markdown because the metadata is the backbone of the system - slide numbers are
what make neighbor expansion (slide x +/- 1) and source citations possible.

Record schema
-------------
{
    "doc_id":      str,   # filename stem, stable id for the document
    "slide_num":   int,   # 1-based page number
    "text":        str,   # slide text + merged note text
    "source_path": str,   # absolute path to the source pdf
    "has_text":    bool,  # did text extraction yield anything?
    "has_notes":   bool,  # were there any annotation/comment notes?
}

The `has_text` flag is the OCR hook for a later version: image-only slides come
back with has_text == False, and OCR can later fill those in by editing only
this stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz  # PyMuPDF


# Notes are merged into the slide text with this marker so that retrieval treats
# the (often very query-relevant) instructor notes as part of the slide.
NOTE_MARKER = "[NOTE]"


def parse_pdf(path: str | Path) -> list[dict]:
    """Parse a single PDF into a list of per-slide records."""
    path = Path(path)
    doc_id = path.stem
    records: list[dict] = []

    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            slide_text = page.get_text("text").strip()

            notes: list[str] = []
            for annot in (page.annots() or []):
                content = (annot.info.get("content") or "").strip()
                if content:
                    notes.append(content)

            combined = slide_text
            if notes:
                note_block = "\n".join(f"{NOTE_MARKER} {n}" for n in notes)
                combined = (combined + "\n" + note_block).strip()

            records.append(
                {
                    "doc_id": doc_id,
                    "slide_num": i + 1,
                    "text": combined,
                    "source_path": str(path.resolve()),
                    "has_text": bool(slide_text),
                    "has_notes": bool(notes),
                }
            )

    return records


def parse_directory(pdf_dir: str | Path, pattern: str = "*.pdf") -> list[dict]:
    """Parse every PDF in a directory (non-recursive) into one record list."""
    pdf_dir = Path(pdf_dir)
    records: list[dict] = []
    for pdf_path in sorted(pdf_dir.glob(pattern)):
        records.extend(parse_pdf(pdf_path))
    return records


def save_records(records: list[dict], out_path: str | Path) -> Path:
    """Write records to a JSONL file (one JSON object per line)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def load_records(in_path: str | Path) -> list[dict]:
    """Read records back from a JSONL file."""
    in_path = Path(in_path)
    with in_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
