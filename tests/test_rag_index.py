import json

import numpy as np

from edge_grader.data import question_id
from edge_grader.rag.index import build, retrieve, top_k, with_ext

VOCAB = ["lever", "fulcrum", "magnet", "iron", "plant", "sunlight", "water", "salt"]


def fake_encode(texts, kind):
    """Bag of words over a tiny vocabulary, normalised: enough to make the right chunk obvious."""
    v = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=np.float32) + 1e-3
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_top_k_orders_by_cosine_and_caps_k():
    vectors = np.eye(3, dtype=np.float32)
    q = np.array([[0.1, 0.9, 0.4]], dtype=np.float32)
    (hits,) = top_k(q, vectors, k=5)
    assert [j for j, _ in hits] == [1, 2, 0]
    assert hits[0][1] > hits[1][1] > hits[2][1]


def test_question_id_keeps_unit_and_question():
    assert question_id("EM.45b.299.1") == "EM.45b"


def test_build_then_retrieve_finds_the_obvious_chunk(tmp_path):
    chunks = [
        {"id": "b:0", "text": "A lever turns about a fulcrum. The fulcrum is the pivot."},
        {"id": "b:1", "text": "A magnet attracts iron. Iron nails stick to a magnet."},
        {"id": "b:2", "text": "A plant needs sunlight and water to make food."},
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in chunks))
    base = build(path, "bge-small-en", fake_encode, out_dir=tmp_path / "index")

    meta = json.loads(with_ext(base, ".json").read_text())
    assert meta["n_chunks"] == 3 and meta["dim"] == len(VOCAB)
    assert np.load(with_ext(base, ".npy")).shape == (3, len(VOCAB))

    qs = {
        "LP.1": {"question": "Where is the fulcrum of this lever?", "splits": ["train"]},
        "ME.2": {"question": "Which objects will the magnet pick up, iron or plastic?",
                 "splits": ["test_ud"]},
    }
    out = retrieve(base, qs, fake_encode, k=2)
    assert out["settings"]["k"] == 2
    assert out["questions"]["LP.1"]["hits"][0][0] == "b:0"
    assert out["questions"]["ME.2"]["hits"][0][0] == "b:1"
    assert out["questions"]["ME.2"]["splits"] == ["test_ud"]
    assert "text" not in json.dumps(out["questions"])  # ids and scores only, never chunk text


def test_check_samples_train_questions_only_and_spreads_units():
    from edge_grader.rag.check import sample_questions

    qs = {f"{u}.{i}": {"splits": ["train", "test_ua"]} for u in ("EM", "LP", "ME") for i in range(5)}
    qs |= {f"EV.{i}": {"splits": ["test_ud"]} for i in range(5)}
    qs |= {"EM.99": {"splits": ["test_uq"]}}
    picked = sample_questions(qs, n=6)
    assert len(picked) == 6
    assert all(qs[q]["splits"][0] == "train" for q in picked)
    assert sorted({q.split(".")[0] for q in picked}) == ["EM", "LP", "ME"]
    assert [q.split(".")[0] for q in picked].count("EM") == 2


def test_two_models_on_one_chunks_file_keep_separate_indexes(tmp_path):
    chunks = tmp_path / "scientsbank-id.jsonl"
    chunks.write_text(json.dumps({"id": "b:0", "text": "lever fulcrum"}) + "\n")
    a = build(chunks, "e5-small-multi", fake_encode, out_dir=tmp_path / "index")
    b = build(chunks, "e5-base-multi", fake_encode, out_dir=tmp_path / "index")
    files = sorted(p.name for p in (tmp_path / "index").iterdir())
    assert files == ["scientsbank-id.e5-base-multi.json", "scientsbank-id.e5-base-multi.npy",
                     "scientsbank-id.e5-small-multi.json", "scientsbank-id.e5-small-multi.npy"]
    assert a != b
