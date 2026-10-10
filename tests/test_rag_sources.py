import hashlib

from edge_grader.rag.sources import BENCHMARKS, fetch, load_manifest, local_path, wiki_filename

MMLU_PRO_SUBJECTS = {
    "biology", "business", "chemistry", "computer_science", "economics", "engineering", "health",
    "history", "law", "math", "other", "philosophy", "physics", "psychology",
}
UNITS = {"EM", "FN", "II", "LF", "LP", "ME", "MS", "MX", "PS", "SE", "ST", "VB", "EV", "HB", "WA"}


def test_manifest_books_are_complete_and_unique():
    books = load_manifest()["books"]
    assert len({b["id"] for b in books}) == len(books)
    for b in books:
        assert b["lang"] in {"en", "id"}
        assert b["url"].startswith("https://")
        assert b["licence"] and b["page_url"]
    assert {b["lang"] for b in books} == {"en", "id"}


def test_books_name_their_benchmark_and_mmlupro_covers_every_subject():
    books = load_manifest()["books"]
    assert {b["benchmark"] for b in books} == set(BENCHMARKS)
    mmlu = [b for b in books if b["benchmark"] == "mmlupro"]
    assert {s for b in mmlu for s in b["subjects"]} == MMLU_PRO_SUBJECTS
    assert {b["lang"] for b in mmlu} == {"en", "id"}


def test_fetch_only_one_benchmark_skips_the_others(tmp_path):
    books = [{"id": "x", "lang": "en", "benchmark": "mmlupro", "url": "https://example.invalid/x.pdf"}]
    assert fetch({"books": books}, tmp_path, only="scientsbank") == []


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


def test_gsm_material_is_one_english_openstax_book():
    gsm = [b for b in load_manifest()["books"] if b["benchmark"] == "gsm"]
    assert [b["id"] for b in gsm] == ["en-openstax-prealgebra-2e"]
    assert gsm[0]["lang"] == "en" and gsm[0]["subjects"] == ["math"]
    assert gsm[0]["sha256"]  # pinned after the first download
