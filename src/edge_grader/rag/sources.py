"""Download the RAG source material listed in sources.json into data/rag/raw/.

    uv run --group rag python -m edge_grader.rag.sources           # download what's missing
    uv run --group rag python -m edge_grader.rag.sources --pin     # also record what was fetched

Books go to data/rag/raw/<lang>/<id>.pdf and are checked against their pinned sha256.
Wikipedia articles go to data/rag/raw/<lang>/wiki/<title>.txt; --pin records their revision ids.
Files already on disk are not downloaded again.
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

MANIFEST = Path(__file__).with_name("sources.json")
RAW_DIR = Path("data/rag/raw")
# Some hosts refuse a default client user agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (edge-grader research; RAG corpus download)"}
# Wikimedia throttles user agents without a contact URL (HTTP 429).
WIKI_HEADERS = {"User-Agent": "edge-grader/0.1 (https://github.com/gnotnek/edge-grader) httpx"}


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text())


def local_path(book: dict, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / book["lang"] / f"{book['id']}.pdf"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    with httpx.stream("GET", url, headers=HEADERS, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    if part.read_bytes()[:5] != b"%PDF-":
        part.unlink()
        raise ValueError(f"not a PDF: {url}")
    part.rename(dest)


def fetch(manifest: dict, raw_dir: Path = RAW_DIR) -> list[dict]:
    """Download missing books and check every book's sha256. Returns one status row per book."""
    rows = []
    for book in manifest["books"]:
        path = local_path(book, raw_dir)
        if not path.exists():
            print(f"downloading {book['id']} ...", flush=True)
            download(book["url"], path)
        digest = sha256_of(path)
        pinned = book.get("sha256")
        status = "new" if pinned is None else ("ok" if digest == pinned else "CHANGED")
        rows.append({"id": book["id"], "sha256": digest, "mb": path.stat().st_size / 1e6, "status": status})
        print(f"{status:8} {book['id']:24} {rows[-1]['mb']:7.1f} MB")
    return rows


def wiki_filename(title: str) -> str:
    return re.sub(r"[^\w()-]+", "_", title).strip("_") + ".txt"


def wiki_query(site: str, **params) -> dict:
    import httpx

    params = {"action": "query", "format": "json", "formatversion": 2, "redirects": 1, **params}
    for attempt in range(6):
        r = httpx.get(f"https://{site}/w/api.php", params=params, headers=WIKI_HEADERS, timeout=60)
        if r.status_code != 429:
            break
        time.sleep(int(r.headers.get("retry-after", 0)) or 5 * 2**attempt)
    r.raise_for_status()
    time.sleep(1)  # stay well under Wikimedia's rate limits
    return r.json()


def wiki_article(site: str, title: str) -> dict | None:
    """Plain text and revision id of one article, following redirects; None if it doesn't exist."""
    d = wiki_query(site, titles=title, prop="extracts|revisions", explaintext=1, rvprop="ids")
    page = d["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    return {
        "site": site,
        "title": page["title"],
        "revid": page["revisions"][0]["revid"],
        "text": page["extract"],
    }


def indonesian_title(en_title: str) -> str | None:
    d = wiki_query("en.wikipedia.org", titles=en_title, prop="langlinks", lllang="id")
    links = d["query"]["pages"][0].get("langlinks") or []
    return links[0]["title"] if links else None


def fetch_wiki(manifest: dict, raw_dir: Path = RAW_DIR) -> dict:
    """Fetch every topic in both languages. Returns {en_title: {units, en, id}} for pinning."""
    wiki = manifest["wikipedia"]
    units: dict[str, list[str]] = {}
    for unit, titles in wiki["topics"].items():
        for t in titles:
            units.setdefault(t, []).append(unit)

    record = {}
    for title, unit_list in units.items():
        pinned = wiki["pinned"].get(title, {})
        entry = {"units": unit_list}
        for lang in ("en", "id"):
            done = pinned.get(lang)
            if done and (raw_dir / lang / "wiki" / wiki_filename(done["title"])).exists():
                entry[lang] = done
                continue
            art = None
            if lang == "en":
                for site in wiki["sites"]["en"]:
                    if art := wiki_article(site, title):
                        break
            elif id_title := indonesian_title(title):
                art = wiki_article(wiki["sites"]["id"][0], id_title)
            if art is None:
                entry[lang] = None
                continue
            path = raw_dir / lang / "wiki" / wiki_filename(art["title"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(art.pop("text"))
            if done and done["revid"] != art["revid"]:
                print(f"note: {lang} {art['title']} changed since pinned (rev {done['revid']} -> {art['revid']})")
            entry[lang] = art
        record[title] = entry
        got = " ".join(f"{lang}={entry[lang]['title'] if entry[lang] else '-'}" for lang in ("en", "id"))
        print(f"wiki {'/'.join(unit_list):6} {title:40} {got}")
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pin", action="store_true", help="write sha256 of new files into sources.json")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    rows = fetch(manifest)
    wiki = fetch_wiki(manifest)
    if args.pin:
        digests = {r["id"]: r["sha256"] for r in rows if r["status"] == "new"}
        for book in manifest["books"]:
            book["sha256"] = digests.get(book["id"], book.get("sha256"))
        manifest["wikipedia"]["pinned"] = wiki
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"pinned {len(digests)} new sha256 and {len(wiki)} Wikipedia topics")
    changed = [r["id"] for r in rows if r["status"] == "CHANGED"]
    if changed:
        raise SystemExit(f"changed upstream since pinning: {', '.join(changed)}")


if __name__ == "__main__":
    main()
