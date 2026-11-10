"""Step 1 of the pipeline: turn files into small text chunks.

Why chunk at all? An embedding represents ONE chunk as ONE vector. A whole
10-K as one vector is a blurry average of everything; an 800-character chunk
is specific enough to match a specific question.
"""
import logging
import re
from pathlib import Path

from pypdf import PdfReader

import config

log = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".txt", ".md"}
_WHITESPACE = re.compile(r"\s")


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(folder=None) -> list[dict]:
    """Read every supported file in `folder`. Empty/unreadable files are
    skipped with a warning instead of crashing the whole run."""
    folder = Path(folder or config.DATA_DIR)
    docs = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        try:
            text = read_file(path).strip()
        except Exception as exc:  # corrupt PDF, encoding problem, ...
            log.warning("Skipping %s: %s", path.name, exc)
            continue
        if not text:
            log.warning("Skipping %s: no extractable text", path.name)
            continue
        docs.append({"source": path.name, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Sliding window over the text.

    - `overlap` repeats the end of one chunk at the start of the next, so a
      sentence cut at a boundary still appears whole in one of the two chunks.
    - We try to end each chunk at a paragraph/sentence/space instead of
      mid-word, which keeps chunks readable and embeddings cleaner.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # look for a natural break in the back half of the window
            window_start = start + chunk_size // 2
            for sep in ("\n\n", ". ", "\n", " "):
                cut = text.rfind(sep, window_start, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        new_start = max(end - overlap, start + 1)  # always move forward
        # Don't begin the next chunk in the middle of a word: skip to the next
        # whitespace so every chunk starts on a whole word.
        if not text[new_start - 1].isspace():
            ws = _WHITESPACE.search(text, new_start)
            if ws and ws.end() < end:
                new_start = ws.end()
        start = new_start
    return chunks


def build_chunks(folder=None, chunk_size=None, overlap=None) -> list[dict]:
    """Documents -> list of {id, source, text} chunk records."""
    records = []
    for doc in load_documents(folder):
        for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            records.append({"id": f"{doc['source']}::{i}", "source": doc["source"], "text": piece})
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    recs = build_chunks()
    print(f"{len(recs)} chunks from {len({r['source'] for r in recs})} documents")
    print("--- sample chunk ---")
    print(recs[0])
