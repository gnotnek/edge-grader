import hashlib

from edge_grader.rag.sources import fetch, load_manifest, local_path, wiki_filename

UNITS = {"EM", "FN", "II", "LF", "LP", "ME", "MS", "MX", "PS", "SE", "ST", "VB", "EV", "HB", "WA"}


def test_manifest_books_are_complete_and_unique():
    books = load_manifest()["books"]
    assert len({b["id"] for b in books}) == len(books)
    for b in books:
        assert b["lang"] in {"en", "id"}
        assert b["url"].startswith("https://")
        assert b["licence"] and b["page_url"]
    assert {b["lang"] for b in books} == {"en", "id"}


def test_manifest_covers_every_scientsbank_unit():
    assert set(load_manifest()["wikipedia"]["topics"]) == UNITS


def test_wiki_filename_is_safe():
    assert wiki_filename("Rock (geology)") == "Rock_(geology).txt"
    assert wiki_filename("Kilau (mineralogi)") == "Kilau_(mineralogi).txt"
    assert "/" not in wiki_filename("AC/DC")


def test_fetch_checks_pinned_sha_without_downloading(tmp_path):
    book = {"id": "b1", "lang": "en", "url": "https://example.invalid/b1.pdf"}
    path = local_path(book, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"%PDF-1.4 test")
    digest = hashlib.sha256(b"%PDF-1.4 test").hexdigest()

    rows = fetch({"books": [{**book, "sha256": digest}]}, tmp_path)
    assert rows[0]["status"] == "ok"
    rows = fetch({"books": [{**book, "sha256": "0" * 64}]}, tmp_path)
    assert rows[0]["status"] == "CHANGED"
    rows = fetch({"books": [{**book, "sha256": None}]}, tmp_path)
    assert rows[0]["status"] == "new"
