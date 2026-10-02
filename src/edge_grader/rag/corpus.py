"""Turn the downloaded RAG material into chunks: extract, clean, chunk (rag.md steps 1-2).

    uv run --group rag python -m edge_grader.rag.corpus                       # SciEntsBank, both langs
    uv run --group rag python -m edge_grader.rag.corpus --benchmark mmlupro

Reads data/rag/raw/<lang>/ (from edge_grader.rag.sources) and writes
data/rag/chunks/<benchmark>-<lang>.jsonl, one chunk per line:
{id, source, lang, benchmark, page_start, page_end, text}. data/ is git-ignored: the
Indonesian books are all rights reserved, so their text must not be committed.

Per book: text blocks from PyMuPDF -> drop image credits and the lines that repeat on
many pages (running headers, footers) -> fix broken ligatures -> drop short blocks
(captions, headings, list options, page numbers) -> 250-word chunks overlapping by 50.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .sources import RAW_DIR, load_manifest, local_path

CHUNK_DIR = Path("data/rag/chunks")
RECORD_DIR = Path("results/rag")

# The Indonesian books' fonts map the fi/fl ligatures to unrelated glyphs. Words where the
# ligature vanished altogether ("Releksi") are rare and can't be told apart safely: left as is.
LIGATURES = {"昀椀": "fi", "昀氀": "fl", "ٴ": "fi", "ٶ": "fi", "˪": "fi"}
LIGATURE_RE = re.compile("|".join(map(re.escape, LIGATURES)))
INVISIBLE_RE = re.compile("[\u200b-‍⁠﻿-]")
BULLETS_RE = re.compile("[●•►▪■□◆◇○◦⦁∙]")
CREDIT_RE = re.compile(
    r"^\s*(image|images|photo|photos|picture|credit|credits|source|sumber|gambar \d)\b"
    r"|pixabay|shutterstock|wikimedia|unsplash|flickr|\bCC0\b|\bCC BY",
    re.IGNORECASE,
)
MIN_BLOCK_WORDS = 8


@dataclass
class Page:
    source: str
    page: int | None  # 1-based PDF page; None for Wikipedia
    blocks: list[str]  # raw text blocks, lines separated by "\n"


def clean_text(text: str) -> str:
    text = LIGATURE_RE.sub(lambda m: LIGATURES[m.group()], text)
    text = unicodedata.normalize("NFKC", text)  # also splits ﬁ/ﬂ
    text = INVISIBLE_RE.sub("", text)
    text = BULLETS_RE.sub(" ", text)
    text = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", text)  # photo-\nsynthesis
    return " ".join(text.split())


def keep_block(text: str) -> bool:
    return len(text.split()) >= MIN_BLOCK_WORDS and not CREDIT_RE.search(text)


def _line_key(line: str) -> str:
    return re.sub(r"[\d\W_]+", " ", line).strip().lower()


def drop_boilerplate(pages: list[Page], min_pages: int = 5, share: float = 0.1) -> list[Page]:
    """Remove lines that recur on many pages of one book (running headers and footers)."""
    seen = Counter()
    for p in pages:
        seen.update({_line_key(line) for b in p.blocks for line in b.splitlines()})
    limit = max(min_pages, share * len(pages))
    common = {k for k, n in seen.items() if n >= limit and k}
    out = []
    for p in pages:
        blocks = []
        for b in p.blocks:
            lines = [line for line in b.splitlines() if _line_key(line) not in common]
            if any(line.strip() for line in lines):
                blocks.append("\n".join(lines))
        out.append(Page(p.source, p.page, blocks))
    return out


def pdf_pages(path: Path, source: str) -> list[Page]:
    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, 1):
            blocks = []
            for *_, text, _, kind in page.get_text("blocks"):
                if kind != 0:  # image block
                    continue
                lines = [line for line in text.splitlines() if not CREDIT_RE.search(line)]
                if lines:
                    blocks.append("\n".join(lines))
            pages.append(Page(source, i, blocks))
    return pages


def wiki_pages(path: Path) -> list[Page]:
    paragraphs = [p for p in path.read_text().split("\n") if p.strip() and not p.startswith("==")]
    return [Page(f"wiki:{path.stem}", None, paragraphs)]


def chunk(pages: list[Page], size: int = 250, overlap: int = 50) -> list[dict]:
    """Fixed word windows over the whole document, so a chunk may span pages."""
    words, page_of = [], []
    for p in pages:
        for b in p.blocks:
            w = clean_text(b).split()
            if len(w) >= MIN_BLOCK_WORDS and keep_block(" ".join(w)):
                words += w
                page_of += [p.page] * len(w)
    source = pages[0].source if pages else ""
    chunks, start, step = [], 0, size - overlap
    while start < len(words) and (start == 0 or start + overlap < len(words)):
        end = min(start + size, len(words))
        chunks.append({
            "id": f"{source}:{len(chunks)}",
            "source": source,
            "page_start": page_of[start],
            "page_end": page_of[end - 1],
            "text": " ".join(words[start:end]),
        })
        start += step
    return chunks


def build(benchmark: str, lang: str, raw_dir: Path = RAW_DIR, size: int = 250,
          overlap: int = 50) -> list[dict]:
    manifest = load_manifest()
    docs = []
    for book in manifest["books"]:
        if book["lang"] != lang or book.get("benchmark") != benchmark:
            continue
        path = local_path(book, raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run python -m edge_grader.rag.sources")
        docs.append(drop_boilerplate(pdf_pages(path, book["id"])))
    if benchmark == "scientsbank":  # the Wikipedia topics fill SciEntsBank's gaps
        docs += [wiki_pages(f) for f in sorted((raw_dir / lang / "wiki").glob("*.txt"))]
    chunks = []
    for pages in docs:
        for c in chunk(pages, size, overlap):
            chunks.append({**c, "lang": lang, "benchmark": benchmark})
    return chunks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--benchmark", default="scientsbank", choices=["scientsbank", "mmlupro"])
    ap.add_argument("--lang", nargs="+", default=["en", "id"])
    ap.add_argument("--size", type=int, default=250, help="words per chunk")
    ap.add_argument("--overlap", type=int, default=50, help="words shared by neighbouring chunks")
    ap.add_argument("--out", type=Path, default=CHUNK_DIR)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    files = {}
    for lang in args.lang:
        chunks = build(args.benchmark, lang, size=args.size, overlap=args.overlap)
        path = args.out / f"{args.benchmark}-{lang}.jsonl"
        with path.open("w") as f:
            f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
        per_source = Counter(c["source"] for c in chunks)
        files[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "chunks": len(chunks),
            "words": sum(len(c["text"].split()) for c in chunks),
            "per_source": dict(sorted(per_source.items())),
        }
        wiki = sum(n for s, n in per_source.items() if s.startswith("wiki:"))
        print(f"{path}: {len(chunks)} chunks ({len(chunks) - wiki} from books, {wiki} from Wikipedia)")
    # The record of what was chunked, committed: chunks are built once here and uploaded to
    # Kaggle as they are, because a fresh Wikipedia download there could differ.
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    record = RECORD_DIR / f"chunks-{args.benchmark}.json"
    record.write_text(json.dumps({
        "stage": "rag",
        "kind": "chunks",
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {"benchmark": args.benchmark, "size_words": args.size,
                     "overlap_words": args.overlap, "min_block_words": MIN_BLOCK_WORDS},
        "files": files,
    }, indent=1) + "\n")
    print(f"wrote {record}")


if __name__ == "__main__":
    main()
