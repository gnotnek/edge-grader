"""Turn the downloaded RAG material into chunks: extract, clean, chunk (rag.md steps 1-2).

    uv run --group rag python -m edge_grader.rag.corpus                       # SciEntsBank, both langs
    uv run --group rag python -m edge_grader.rag.corpus --benchmark mmlupro
    uv run python -m edge_grader.rag.corpus --benchmark news --lang en       # LiveNewsBench articles

Reads data/rag/raw/<lang>/ (from edge_grader.rag.sources) and writes
data/rag/chunks/<benchmark>-<lang>.jsonl, one chunk per line:
{id, source, lang, benchmark, page_start, page_end, text}. data/ is git-ignored: the
Indonesian books are all rights reserved, so their text must not be committed.

Per book: text blocks from PyMuPDF -> cut front and back matter (contents, glossary,
bibliography, author profiles; for OpenStax, everything outside Chapter 1 .. the first appendix,
answer key, reference list or index, by the PDF's bookmarks) -> drop image credits and the lines that repeat on many pages
(running headers, footers) -> fix broken ligatures -> drop blocks that aren't course material:
short ones (captions, headings, list options, page numbers), references, worksheet prompts, and
ones mostly addressed to the student (questions, activity steps) -> 250-word chunks
overlapping by 50.

News (the topic benchmark): LiveNewsBench's source articles, one document per distinct article
(source "news:<id>", see edge_grader.news.article_id), each paragraph a block, then the same
filters and windows. Some articles aren't in English; they're kept, as the benchmark gives them.
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
INVISIBLE_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff\ue000-\uf8ff]")  # incl. private use
BULLETS_RE = re.compile(r"[\u25cf\u2022\u25ba\u25aa\u25a0\u25a1\u25c6\u25c7\u25cb\u25e6\u2981\u2219]")
CREDIT_RE = re.compile(
    r"^\s*(image|images|photo|photos|picture|credit|credits|source|sumber|gambar \d)\b"
    r"|pixabay|shutterstock|wikimedia|unsplash|flickr|\bCC0\b|\bCC BY",
    re.IGNORECASE,
)
# Text that reads as course material for nobody: reference lists, the books' fixed worksheet
# prompts ("What are you observing about this phenomenon?") and standards codes ("3.2.2 Analyze").
CITATION_RE = re.compile(r"https?://|www\.|\bdiakses\b|\baccessed\b|retrieved from|\bdoi:|\bISBN\b",
                         re.IGNORECASE)
WORKSHEET_RE = re.compile(
    r"pelajar pancasila|buku (tugas|catatan)|what are you (observing|wondering)|^\d\.\d\.\d\b",
    re.IGNORECASE,
)
# A block is mostly addressed to the student (questions, activity steps) when at least half its
# sentences are questions or start with a command. Specific verbs only: a loose pattern would
# also catch "Jumlah ..." (amount) or "Work is ...".
COMMANDS_ID = (
    r"ayo|mari|(tulis|buat|cari|ambil|baca|isi|lakukan|perhatikan|jawab|gambar|susun|selidiki"
    r"|diskusikan|sebutkan|jelaskan|amati|catat|ceritakan|tentukan|lengkapi|salin|pilih|hitung"
    r"|uji|siapkan|gunakan|bandingkan|kumpulkan|simpulkan|rancang|coba|tanyakan|bayangkan"
    r"|perkirakan|presentasikan|tuliskan|ulangi|identifikasi|analisislah)(lah)?"
)
COMMANDS_EN = (
    r"explain|describe|write|draw|read|discuss|compare|identify|complete|circle|think|share"
    r"|predict|create|observe|ask|develop|construct|obtain|evaluate|analyze|carry|try|find"
    r"|choose|imagine|talk|make|look"
)
COMMAND_RE = re.compile(rf"^(\d+[.)]\s*|[a-z][.)]\s*)?({COMMANDS_ID}|{COMMANDS_EN})\b", re.IGNORECASE)
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


def directive_share(text: str) -> float:
    """Share of sentences that are questions or commands to the student."""
    sentences = [s for s in re.split(r"(?<=[.?!])\s+", text) if len(s.split()) >= 2]
    if not sentences:
        return 0.0
    return sum(s.rstrip().endswith("?") or bool(COMMAND_RE.match(s)) for s in sentences) / len(
        sentences
    )


def keep_block(text: str) -> bool:
    return (
        len(text.split()) >= MIN_BLOCK_WORDS
        and not CREDIT_RE.search(text)
        and not CITATION_RE.search(text)
        and not WORKSHEET_RE.search(text)
        and directive_share(text) < 0.5
    )


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


FRONT_RE = re.compile(
    r"^\s*(kata pengantar|daftar isi|daftar gambar|daftar tabel|petunjuk penggunaan buku)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
BACK_RE = re.compile(
    r"^\s*(daftar pustaka|glosarium|indeks"
    r"|profil (penulis|penelaah|ilustrator|editor|desainer|pelaku perbukuan))\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def body_pages(pages: list[Page]) -> list[Page]:
    """Cut the front matter (preface, contents, how to use the book) and the back matter
    (glossary, bibliography, index, author profiles). Headings count only in the first 10% or
    last 25% of the book, so a table of contents listing "Glosarium" doesn't cut it short."""
    n = len(pages)
    front, back = 0, n + 1
    for p in pages:
        text = "\n".join(p.blocks)
        if p.page <= 0.1 * n and FRONT_RE.search(text):
            front = p.page
        if p.page >= 0.75 * n and back > n and BACK_RE.search(text):
            back = p.page
    return [p for p in pages if front < p.page < back]


# OpenStax books carry a bookmark outline: Contents, Preface, Chapter 1 ... Chapter N (or Unit),
# then Appendix, Answer Key, References, Index. Their headings don't match the Indonesian
# patterns above, so for them the outline sets the body. It isn't used for the other books, so
# their chunks (and the sha256s the Kaggle notebooks check) don't change.
CHAPTER_RE = re.compile(r"^(chapter|unit)\s+\d", re.IGNORECASE)
BACK_TOC_RE = re.compile(r"^(appendix|answer key|references|index)\b", re.IGNORECASE)


def toc_body(toc: list[list]) -> tuple[int, int] | None:
    """First and last body page from a PDF's top-level bookmarks: the first chapter (or unit)
    up to the page before the first appendix, answer key, reference list or index."""
    top = [(title.strip(), page) for level, title, page, *_ in toc if level == 1]
    starts = [page for title, page in top if CHAPTER_RE.match(title)]
    if not starts:
        return None
    first = min(starts)
    backs = [page for title, page in top if BACK_TOC_RE.match(title) and page > first]
    return first, (min(backs) - 1 if backs else max(page for _, page in top))


def openstax_body(path: Path, pages: list[Page]) -> list[Page]:
    import pymupdf

    with pymupdf.open(path) as doc:
        span = toc_body(doc.get_toc())
    if span is None:
        raise ValueError(f"{path}: no chapter bookmarks; can't find the body")
    return [p for p in pages if span[0] <= p.page <= span[1]]


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


def news_pages() -> list[list[Page]]:
    """Each distinct LiveNewsBench article as one document, in first-seen order."""
    from ..news import article_id, load

    docs, seen = [], set()
    for q in load():
        for a in q["articles"]:
            aid = article_id(a)
            if aid not in seen:
                seen.add(aid)
                paragraphs = [p for p in a["article_content"].split("\n\n") if p.strip()]
                docs.append([Page(f"news:{aid}", None, paragraphs)])
    return docs


def build(benchmark: str, lang: str, raw_dir: Path = RAW_DIR, size: int = 250,
          overlap: int = 50) -> list[dict]:
    if benchmark == "news":
        docs = news_pages()
        return [{**c, "lang": lang, "benchmark": benchmark}
                for pages in docs for c in chunk(pages, size, overlap)]
    manifest = load_manifest()
    docs = []
    for book in manifest["books"]:
        if book["lang"] != lang or book.get("benchmark") != benchmark:
            continue
        path = local_path(book, raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run python -m edge_grader.rag.sources")
        pages = pdf_pages(path, book["id"])
        body = openstax_body(path, pages) if book["id"].startswith("en-openstax-") else body_pages(pages)
        docs.append(drop_boilerplate(body))
    if benchmark == "scientsbank":  # the Wikipedia topics fill SciEntsBank's gaps
        docs += [wiki_pages(f) for f in sorted((raw_dir / lang / "wiki").glob("*.txt"))]
    chunks = []
    for pages in docs:
        for c in chunk(pages, size, overlap):
            chunks.append({**c, "lang": lang, "benchmark": benchmark})
    return chunks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--benchmark", default="scientsbank", choices=["scientsbank", "mmlupro", "news"])
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
        if args.benchmark == "news":
            del files[path.name]["per_source"]  # one entry per article: too long to be useful
            files[path.name]["articles"] = len(per_source)
            print(f"{path}: {len(chunks)} chunks from {len(per_source)} articles")
            continue
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
