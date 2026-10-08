# Phase C: text + images for grading, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- **C1:** grade SciEntsBank answers given **as an image** (each student answer rendered in a handwriting-style font), paired answer by answer with the text results.
- **Reading check:** separate reading errors from grading errors.
- **Cost:** measure it on the boards.
- **C2:** find out whether a dataset of real student answers to questions **with figures** exists, and run it only if one qualifies.

**Architecture:**
- **`edge_grader/images.py`:**
  - renders answer text to PNG deterministically, with a pinned font and Pillow version
  - writes a committed manifest of image sha256s
  - holds the reading check (transcription + character error rate)
- **The grader** sends the unchanged prompt as a multi-part message. The image sits exactly where the student answer's text was.
- **Serving:** `llama-server` with the baseline flags plus `--mmproj <Gemma 4 vision projector>`.
- **Runs:**
  - Kaggle: E2B + E4B, test_ua + test_uq
  - Jetson: test_ua cost
  - Pi: a 100-answer cost sample

**Tech Stack:** Python 3 via uv, Pillow (new `images` dependency group), pytest, llama.cpp multimodal (`--mmproj`), Kaggle notebook, bash board script.

**Spec:** [`docs/superpowers/specs/2026-10-08-baseline-variations-design.md`](../specs/2026-10-08-baseline-variations-design.md) §7. **Depends on:** phase 0 (board script and board-check) merged.

## Global Constraints

**Tooling:** Python via uv only (`uv add --group images pillow`). The Mac renders images but never runs Gemma.

**Prompt:**
- `grader.build_prompt` text is unchanged except one line: `Student answer: (in the image below)`.
- The image follows that line; the question and reference answer stay text.
- Greedy, `max_tokens` 16, the same parser.

**Serving:**
- the baseline flags (`-c 8192 --host 127.0.0.1 --port 8080 -rea off --reasoning-budget -1 --cache-ram 0`, `-ngl 99` Kaggle/Jetson, Pi `-ngl 0 -t 3`), **plus `--mmproj <file>` only**
- the projector must match the QAT GGUFs; if it can't, that's recorded as a confound (spec §10)

**Images:**
- built on the Mac
- reproducible from the dataset, the pinned font (sha256) and the pinned Pillow version
- the images themselves stay in `data/` (git-ignored); a manifest of sha256s is committed
- uploaded to Kaggle as a **private** dataset, and checked against the manifest before grading

**Metrics:** accuracy and macro-F1; McNemar paired with the text run on the same answer ids; character error rate for the reading check.

**Results and separation:**
- every run writes its summary JSON to `results/`; never hand-edit
- board runs only through `scripts/board_grade.sh`

**C2:** no self-made dataset. Real student answers, a figure in the question, grade labels, English, a licence that allows research use, roughly 300+ answers; otherwise C2 is dropped and listed as a gap.

**Commits:** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **The same answer must render to the same bytes every time,** or the manifest check on Kaggle fails and runs can't be repeated. Task 2: `test_render_is_deterministic`.
2. **A long answer must wrap, not run off the image or get cut.** Task 2: `test_long_answers_wrap_inside_the_width`.
3. **An image file that doesn't match the manifest** (re-rendered with another font or Pillow) must stop the run before any grading. Task 3: `test_image_run_refuses_a_changed_image`.
4. **The image must replace the answer text, not be added to it.** If the answer text leaks into the text part, the "image" condition is really text. Task 3: `test_image_message_has_no_answer_text`.
5. **A server started without `--mmproj`** returns an error or ignores the image. A whole run of errors or random grades must be visible. The runner counts errors as now, and the pilot checks the reading-check transcriptions resemble the answers. Task 4, Step 3.

---

## File map

| File | Change | Responsibility |
|---|---|---|
| `pyproject.toml`, `uv.lock` | Modify | `images` group: Pillow |
| `src/edge_grader/images.py` | Create | Font fetch, render, manifest, image lookup, CER, reading check CLI |
| `src/edge_grader/grader.py` | Modify | `image_content()`; `grade(..., image=None)` |
| `src/edge_grader/baseline.py` | Modify | `--images <dir>`: `img-*` runs |
| `tests/test_images.py` | Create | Rendering, manifest, CER |
| `tests/test_grader.py`, `tests/test_baseline_e2e.py` | Modify | Image prompt and run |
| `notebooks/kaggle_baseline.ipynb` | Modify | `IMAGES` switch, projector download, images input |
| `scripts/board_grade.sh`, `tests/test_board_grade.py` | Modify | `IMAGES=<dir>` → `--mmproj` and `--images` |
| `src/edge_grader/board_check.py` | Modify | `img-*` summaries (via `SUMMARY_GLOBS` and the stage prefix) |
| `docs/results/images.md` | Create | C1 and C2 results |

---

### Task 1: find the vision projector

**Files:** none. A decision row in `docs/research-outline.md`.

- [ ] **Step 1: List the projector files next to the lab's GGUFs** (Mac; a file listing, no download)

```bash
uv run python -c "
from huggingface_hub import list_repo_files
for repo in ('unsloth/gemma-4-E2B-it-qat-GGUF', 'unsloth/gemma-4-E4B-it-qat-GGUF'):
    print(repo, [f for f in list_repo_files(repo) if 'mmproj' in f.lower()])"
```

**Expected:** at least one `mmproj-*.gguf` per repo (e.g. `mmproj-F16.gguf`).
- **Repo has one:** use it. Same publisher and the same QAT release, so it's matched.
- **Repo has none:** look in `ggml-org/` and `google/` Gemma 4 E2B/E4B GGUF repos the same way, and use the projector from the release whose base weights match. Record the mismatch as a confound.
- **Nothing found anywhere:** stop C1 and tell the owner.

- [ ] **Step 2: Record**

Add an own-decisions row to `docs/research-outline.md`: the projector repo and file per model, and why it matches (or the confound). Commit.

---

### Task 2: render answers to images

**Files:**
- Create: `src/edge_grader/images.py`, `tests/test_images.py`
- Modify: `pyproject.toml` (group `images`; script `edge-grader-render`)

**Interfaces:**
- Produces:
  - `images.FONT = {"url": str, "sha256": str, "licence": str, "file": "PatrickHand-Regular.ttf"}`
  - `images.font_path(fonts_dir: Path = FONTS_DIR) -> Path`: downloads once and checks the sha256
  - `images.render(text: str, font: Path, width: int = 768, size: int = 32, margin: int = 24) -> bytes` (PNG)
  - `images.image_path(images_dir: Path, split: str, answer_id: str) -> Path` = `images_dir / split / f"{answer_id}.png"`
  - `images.load_manifest(images_dir: Path, split: str) -> dict[str, str]` (id → sha256)
  - CLI `edge-grader-render --split test_ua [test_uq …] --out data/images`: writes the PNGs and `results/images/manifest-<split>.json`
  - `images.cer(hyp: str, ref: str) -> float`

- [ ] **Step 1: Add Pillow and choose the font**

```bash
uv add --group images pillow
```

**Font:** **Patrick Hand** (SIL Open Font License), a plain handwriting style that stays readable.
- **URL:** from the `google/fonts` GitHub repository, `ofl/patrickhand/PatrickHand-Regular.ttf` via `https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf`.
- **Licence:** check that the same folder has `OFL.txt`.
- **Pinning:** the sha256 is taken at first download (Step 4) and written into `FONT`, like the RAG books.
- If the file isn't at that path, find it in the same repository; don't use a font from another source without telling the owner.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_images.py`:

```python
"""Answer images: deterministic rendering, wrapping, the manifest, character error rate."""

import io

import pytest

PIL = pytest.importorskip("PIL")  # the images group isn't installed everywhere (boards, Kaggle grading)
from PIL import Image  # noqa: E402

from edge_grader import images  # noqa: E402


@pytest.fixture(scope="module")
def font():
    try:
        return images.font_path()
    except Exception as e:  # no network: skip rather than fail
        pytest.skip(f"font unavailable: {e}")


def test_render_is_deterministic(font):
    a = images.render("Because the magnet attracts iron.", font)
    assert a == images.render("Because the magnet attracts iron.", font)
    assert a[:8] == b"\x89PNG\r\n\x1a\n"


def test_long_answers_wrap_inside_the_width(font):
    short = Image.open(io.BytesIO(images.render("Yes.", font)))
    long = Image.open(io.BytesIO(images.render("word " * 120, font)))
    assert short.width == long.width == 768
    assert long.height > 4 * short.height  # wrapped onto many lines, not cut


def test_manifest_round_trip(tmp_path, font):
    out = tmp_path / "img"
    images.write_split(out, "test_ua", {"EM.1.1": "Yes.", "EM.1.2": "No."}, font,
                       manifest_dir=tmp_path / "results")
    m = images.load_manifest(tmp_path / "results", "test_ua")
    assert set(m) == {"EM.1.1", "EM.1.2"}
    assert images.image_path(out, "test_ua", "EM.1.1").exists()


def test_cer():
    assert images.cer("the cat", "the cat") == 0.0
    assert images.cer("the bat", "the cat") == pytest.approx(1 / 7)
    assert images.cer("", "abc") == 1.0
    assert images.cer("  the   cat ", "the cat") == 0.0  # whitespace normalised
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run --group images pytest tests/test_images.py -v`
Expected: `ImportError: cannot import name 'images'`.

- [ ] **Step 4: Implement `images.py`**

```python
"""Student answers as images (phase C1): render each SciEntsBank answer in a handwriting-style
font, deterministically, and keep a manifest of the images' sha256 so a run can check it grades
exactly these images. The images stay in data/ (git-ignored); the manifest is committed.

    uv run --group images edge-grader-render --split test_ua test_uq       # data/images/<split>/<id>.png

Also the reading check's measure, the character error rate (cer).
"""

import argparse
import hashlib
import io
import json
import textwrap
from datetime import UTC, datetime
from pathlib import Path

FONTS_DIR = Path("data/images/fonts")
IMAGES_DIR = Path("data/images")
MANIFEST_DIR = Path("results/images")
FONT = {
    "file": "PatrickHand-Regular.ttf",
    "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    "licence": "SIL Open Font License 1.1 (google/fonts ofl/patrickhand/OFL.txt)",
    "sha256": None,  # pinned after the first download: Step 4 of the phase C plan
}


def font_path(fonts_dir: Path = FONTS_DIR) -> Path:
    import httpx

    path = fonts_dir / FONT["file"]
    if not path.exists():
        fonts_dir.mkdir(parents=True, exist_ok=True)
        r = httpx.get(FONT["url"], follow_redirects=True, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if FONT["sha256"] and digest != FONT["sha256"]:
        raise SystemExit(f"{path}: sha256 {digest[:12]} isn't the pinned font; delete it and rerun")
    return path


def render(text: str, font: Path, width: int = 768, size: int = 32, margin: int = 24) -> bytes:
    """Black text on white, wrapped to the width; PNG bytes with no timestamp or metadata."""
    from PIL import Image, ImageDraw, ImageFont

    face = ImageFont.truetype(str(font), size)
    avg = face.getlength("abcdefghijklmnopqrstuvwxyz") / 26
    lines = textwrap.wrap(" ".join(text.split()), width=max(1, int((width - 2 * margin) / avg))) or [""]
    line_h = int(size * 1.4)
    img = Image.new("L", (width, 2 * margin + line_h * len(lines)), 255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((margin, margin + i * line_h), line, font=face, fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def image_path(images_dir: Path, split: str, answer_id: str) -> Path:
    return images_dir / split / f"{answer_id}.png"


def write_split(images_dir: Path, split: str, answers: dict[str, str], font: Path,
                manifest_dir: Path = MANIFEST_DIR) -> Path:
    from PIL import __version__ as pillow_version

    (images_dir / split).mkdir(parents=True, exist_ok=True)
    digests = {}
    for answer_id, text in sorted(answers.items()):
        png = render(text, font)
        image_path(images_dir, split, answer_id).write_bytes(png)
        digests[answer_id] = hashlib.sha256(png).hexdigest()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"manifest-{split}.json"
    out.write_text(json.dumps({
        "stage": "images", "kind": "manifest", "split": split,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {"font": FONT["file"], "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
                     "pillow": pillow_version, "width": 768, "size": 32, "margin": 24},
        "images": digests,
    }, indent=1) + "\n")
    return out


def load_manifest(manifest_dir: Path, split: str) -> dict[str, str]:
    return json.loads((manifest_dir / f"manifest-{split}.json").read_text())["images"]


def cer(hyp: str, ref: str) -> float:
    """Character error rate: edit distance over the reference length, whitespace normalised."""
    a, b = " ".join(hyp.split()), " ".join(ref.split())
    if not b:
        return 0.0 if not a else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return min(1.0, prev[-1] / len(b))


def main(argv: list[str] | None = None) -> None:
    from . import data

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", nargs="+", default=["test_ua"], choices=["test_ua", "test_uq", "test_ud"])
    ap.add_argument("--out", type=Path, default=IMAGES_DIR)
    args = ap.parse_args(argv)
    font = font_path()
    for split in args.split:
        answers = {i.id: i.student_answer for i in data.load(split)}
        print(f"{split}: {len(answers)} answers -> {write_split(args.out, split, answers, font)}")


if __name__ == "__main__":
    main()
```

Register the script in `pyproject.toml` under `[project.scripts]`: `edge-grader-render = "edge_grader.images:main"`.

Then pin the font. Run once: `uv run --group images python -c "from edge_grader.images import font_path; import hashlib; p=font_path(); print(hashlib.sha256(p.read_bytes()).hexdigest())"`. Put that digest into `FONT["sha256"]`, and check `OFL.txt` exists in the same `google/fonts` folder.

- [ ] **Step 5: Run the tests**

Run: `uv run --group images pytest tests/test_images.py -v && uv run pytest -q && uv run ruff check .`
Expected:
- the image tests PASS
- in the default environment (without the group), `tests/test_images.py` is skipped, not failed

- [ ] **Step 6: Render the splits and look at a few images**

```bash
uv run --group images edge-grader-render --split test_ua test_uq
open data/images/test_ua/$(ls data/images/test_ua | head -1)
```

**The owner looks at 5 images** and confirms they read as handwriting and aren't cut off.

Commit the code and manifests (not the images):

```bash
git add pyproject.toml uv.lock src/edge_grader/images.py tests/test_images.py results/images/manifest-test_ua.json results/images/manifest-test_uq.json
git commit -m "C1: render SciEntsBank answers as handwriting images; committed sha256 manifests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: grade from an image

**Files:**
- Modify: `src/edge_grader/grader.py`, `src/edge_grader/baseline.py`
- Test: `tests/test_grader.py`, `tests/test_baseline_e2e.py`

**Interfaces:**
- Consumes: Task 2's `images.image_path`, `images.load_manifest`.
- Produces:
  - `grader.IMAGE_ANSWER = "(in the image below)"`
  - `grader.image_content(item, labels, png: bytes) -> list[dict]`: OpenAI content parts `[text, image_url, text]`
  - `grade(..., image: bytes | None = None)`
  - `baseline --images <dir> [--manifests results/images]`:
    - checks every image's sha256 against the manifest before the first request
    - stage `"img"`, files `img-<label>-<split>-<scheme>-<stamp>`
    - records `"images": {"dir", "manifest", "font_sha256", "pillow"}`
    - refuses `--rag`/`--thinking` together with `--images`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_grader.py`:

```python
from edge_grader.grader import IMAGE_ANSWER, image_content


def test_image_message_has_no_answer_text():
    item = Item("1", "Why is the sky blue?", "Scattering.", "Because of scattering.", "correct")
    parts = image_content(item, LABELS, b"\x89PNGfake")
    text = "".join(p["text"] for p in parts if p["type"] == "text")
    assert "Because of scattering." not in text
    assert f"Student answer: {IMAGE_ANSWER}" in text
    assert [p["type"] for p in parts] == ["text", "image_url", "text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # the text around the image is the stage-1 prompt with only the answer line changed
    with_marker = build_prompt(Item("1", item.question, item.reference_answer, IMAGE_ANSWER, "correct"), LABELS)
    assert parts[0]["text"] + parts[2]["text"] == with_marker
```

In `tests/test_baseline_e2e.py`, make the fake server handle content lists. In `do_POST`, replace `prompt = body["messages"][0]["content"]` with:

```python
        content = body["messages"][0]["content"]
        if isinstance(content, list):  # an image request: "right" answers are images named a.png
            prompt = "".join(p.get("text", "") for p in content)
            img = next(p for p in content if p["type"] == "image_url")["image_url"]["url"]
            prompt += "Student answer: right" if img.endswith(RIGHT_B64) else ""
        else:
            prompt = content
```

and define near the top: `RIGHT_PNG = b"\x89PNG-right"`, `RIGHT_B64 = __import__("base64").b64encode(RIGHT_PNG).decode()`. Then append:

```python
def write_images(tmp_path, items, tamper=False):
    import hashlib
    d, m = tmp_path / "img", tmp_path / "manifests"
    (d / "test_ua").mkdir(parents=True)
    m.mkdir()
    digests = {}
    for it in items:
        png = RIGHT_PNG if it.student_answer.startswith("right") else b"\x89PNG-other-" + it.id.encode()
        (d / "test_ua" / f"{it.id}.png").write_bytes(png)
        digests[it.id] = hashlib.sha256(png).hexdigest()
    if tamper:
        (d / "test_ua" / f"{items[0].id}.png").write_bytes(b"changed")
    (m / "manifest-test_ua.json").write_text(json.dumps(
        {"settings": {"font_sha256": "f", "pillow": "p"}, "images": digests}))
    return d, m


def test_image_run_grades_from_images(server, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load", lambda *a, **k: ITEMS)
    d, m = write_images(tmp_path, ITEMS)
    baseline.main(["--base-url", server, "--label", "fake", "--n", "0", "--out", str(tmp_path / "out"),
                   "--images", str(d), "--manifests", str(m)])
    (f,) = (tmp_path / "out").glob("img-fake-test_ua-3way-*.json")
    run = json.loads(f.read_text())
    assert run["stage"] == "img" and run["metrics"]["accuracy"] == 2 / 3
    assert run["images"]["font_sha256"] == "f"


def test_image_run_refuses_a_changed_image(server, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load", lambda *a, **k: ITEMS)
    d, m = write_images(tmp_path, ITEMS, tamper=True)
    with pytest.raises(SystemExit, match="sha256"):
        baseline.main(["--base-url", server, "--n", "0", "--out", str(tmp_path / "out"),
                       "--images", str(d), "--manifests", str(m)])
```

The fake grades "right" answers correct and others incorrect. For the image run, the fake marks only `RIGHT_PNG` as right, so the accuracy is again 2/3, as in the text test.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_grader.py tests/test_baseline_e2e.py -v`
Expected: ImportError on `IMAGE_ANSWER`, and `--images` unrecognised.

- [ ] **Step 3: Implement `grader.py`**

```python
IMAGE_ANSWER = "(in the image below)"


def image_content(item: Item, labels: list[str], png: bytes) -> list[dict]:
    """The stage-1 prompt with the answer replaced by an image, placed right after its line."""
    import base64
    from dataclasses import replace

    text = build_prompt(replace(item, student_answer=IMAGE_ANSWER), labels)
    cut = text.index(IMAGE_ANSWER) + len(IMAGE_ANSWER)
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return [{"type": "text", "text": text[:cut]},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text[cut:]}]
```

In `grade`, add the parameter `image: bytes | None = None`, and build the message content as `content = image_content(item, labels, image) if image is not None else build_prompt(item, labels, passages)`. Send `messages=[{"role": "user", "content": content}]`. Keep everything else, including B1's `thinking` handling if it's merged.

- [ ] **Step 4: Implement `baseline.py`**

1. Add the arguments `--images` (a directory) and `--manifests` (default `results/images`).
2. After parsing: `if args.images and (args.rag or getattr(args, "thinking", False)): sys.exit("--images runs alone: no --rag or --thinking")`.
3. `stage = "img"` when `--images` is set (keeping B1's think/rag order: `think` > `rag` > `img` > `baseline`; they're exclusive anyway).
4. After loading `items`, if `--images`:

```python
        import hashlib

        from . import images as img
        manifest = img.load_manifest(Path(args.manifests), args.split)
        pngs = {}
        for item in items:
            p = img.image_path(Path(args.images), args.split, item.id)
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != manifest.get(item.id):
                sys.exit(f"{p}: missing, or its sha256 isn't the manifest's; re-render or re-upload")
            pngs[item.id] = p.read_bytes()
        mf = json.loads((Path(args.manifests) / f"manifest-{args.split}.json").read_text())["settings"]
```

`images.py` imports Pillow only inside `render`/`write_split`, so `load_manifest`/`image_path` work without the `images` group (Kaggle and the boards don't install it).

5. Pass `image=pngs.get(item.id)` to `grader.grade`.
6. Record `run["images"] = {"dir": args.images, "manifest": f"{args.manifests}/manifest-{args.split}.json", "font_sha256": mf["font_sha256"], "pillow": mf["pillow"]}`.

- [ ] **Step 5: Tests, lint, commit**

Run: `uv run pytest -q && uv run ruff check .`

```bash
git add src/edge_grader/grader.py src/edge_grader/baseline.py tests/test_grader.py tests/test_baseline_e2e.py
git commit -m "C1: grade from an answer image (img-* runs, manifest-checked)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: the reading check and the Kaggle runs

**Files:**
- Modify: `src/edge_grader/images.py` (a `read-check` command), `notebooks/kaggle_baseline.ipynb`
- Test: `tests/test_images.py`

**Interfaces:**
- Produces: `edge-grader-read-check --images <dir> --split test_ua --n 100 --seed 0 --out results/kaggle`, which writes `read-<label>-<split>-<stamp>.json` (median and mean CER, the share of answers with CER ≤ 0.1) and `raw/…jsonl` (`id`, transcription, cer).

- [ ] **Step 1: The reading check command**

Append to `images.py`:

```python
READ_PROMPT = "Transcribe the handwritten text in this image exactly. Reply with the text only."


def read_main(argv: list[str] | None = None) -> None:
    """The reading check: transcribe n answer images and score them against the dataset text."""
    import base64
    import random
    import statistics

    from openai import OpenAI

    from . import data
    from .baseline import server_info

    ap = argparse.ArgumentParser(description=read_main.__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--label", default="local")
    ap.add_argument("--images", type=Path, default=IMAGES_DIR)
    ap.add_argument("--manifests", type=Path, default=MANIFEST_DIR)
    ap.add_argument("--split", default="test_ua")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/local"))
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifests, args.split)
    items = sorted(data.load(args.split), key=lambda i: i.id)
    picked = random.Random(args.seed).sample(items, min(args.n, len(items)))
    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=600, max_retries=3)
    server, rows = server_info(args.base_url), []
    for it in picked:
        png = image_path(args.images, args.split, it.id).read_bytes()
        if hashlib.sha256(png).hexdigest() != manifest[it.id]:
            raise SystemExit(f"{it.id}: image doesn't match the manifest")
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        r = client.chat.completions.create(model="local", temperature=0, max_tokens=256, messages=[
            {"role": "user", "content": [{"type": "text", "text": READ_PROMPT},
                                         {"type": "image_url", "image_url": {"url": url}}]}])
        text = r.choices[0].message.content or ""
        rows.append({"id": it.id, "transcription": text, "cer": round(cer(text, it.student_answer), 4)})
    cers = [r["cer"] for r in rows]
    now = datetime.now(UTC)
    stem = f"read-{args.label}-{args.split}-{now:%Y%m%d-%H%M%S}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw").mkdir(exist_ok=True)
    (args.out / f"{stem}.json").write_text(json.dumps({
        "stage": "img", "kind": "read-check", "label": args.label, "split": args.split,
        "server": server, "time": now.isoformat(timespec="seconds"),
        "settings": {"n": len(rows), "seed": args.seed, "prompt": READ_PROMPT},
        "metrics": {"n": len(rows), "median_cer": statistics.median(cers), "mean_cer": round(statistics.fmean(cers), 4),
                    "share_cer_le_0.1": round(sum(c <= 0.1 for c in cers) / len(cers), 4)},
    }, indent=2))
    (args.out / "raw" / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"median CER {statistics.median(cers):.3f} over {len(rows)} -> {args.out / (stem + '.json')}")
```

Register `edge-grader-read-check = "edge_grader.images:read_main"` in `pyproject.toml`. Append to `tests/test_images.py`. This test needs no Pillow, so put it **above** the `importorskip` line, or in a new file `tests/test_read_check.py`, which is better:

```python
"""The reading check against a fake server that transcribes perfectly."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from edge_grader import data, images

ITEMS = [data.Item("a", "Q?", "ref", "magnets pull iron", "correct"),
         data.Item("b", "Q?", "ref", "it floats", "incorrect")]


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({"model_path": "/m.gguf", "default_generation_settings": {"n_ctx": 8192}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        url = body["messages"][0]["content"][1]["image_url"]["url"]
        text = next(i.student_answer for i in ITEMS if url.endswith(
            __import__("base64").b64encode(f"png-{i.id}".encode()).decode()))
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": "local",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 300, "completion_tokens": 5, "total_tokens": 305}})


def test_read_check_scores_a_perfect_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load", lambda *a, **k: ITEMS)
    d, m = tmp_path / "img", tmp_path / "man"
    (d / "test_ua").mkdir(parents=True)
    m.mkdir()
    digests = {}
    for it in ITEMS:
        png = f"png-{it.id}".encode()
        (d / "test_ua" / f"{it.id}.png").write_bytes(png)
        digests[it.id] = hashlib.sha256(png).hexdigest()
    (m / "manifest-test_ua.json").write_text(json.dumps({"images": digests}))
    httpd = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        images.read_main(["--base-url", f"http://127.0.0.1:{httpd.server_port}/v1", "--label", "fake",
                          "--images", str(d), "--manifests", str(m), "--n", "2", "--out", str(tmp_path / "out")])
    finally:
        httpd.shutdown()
    (f,) = (tmp_path / "out").glob("read-fake-test_ua-*.json")
    assert json.loads(f.read_text())["metrics"]["median_cer"] == 0.0
```

Run `uv run pytest tests/test_read_check.py -v` (it should pass), then the full suite, then commit.

- [ ] **Step 2: Kaggle notebook switch** (edit script, as in phase A Task 5)

**`c01`:**
- add `IMAGES = False` and `READ_CHECK_N = 100` under a `# Phase C1` comment
- add `MMPROJ = {"e2b": (<repo>, <file>), "e4b": (<repo>, <file>)}`, filled from Task 1's record

**`c06`:** if `IMAGES`, also download each projector with `hf_hub_download` into `/tmp/models`, as `MMPROJ_PATHS`.

**The RAG-chunks cell, or a new cell after it:** if `IMAGES`, for each split in `SPLITS`, find the folder in the private images dataset with `src = next(Path("/kaggle/input").rglob(split))` and copy it to `/kaggle/working/edge-grader/data/images/<split>` with `shutil.copytree(src, dest, dirs_exist_ok=True)`. If it isn't found, raise the same "add the dataset as an input" error as the chunks cell. The runner itself checks every image's sha256 against the committed manifest, so the notebook only copies.

**`c11`:**
- `start_server` appends `["--mmproj", MMPROJ_PATHS[key]]` when `IMAGES`
- the SciEntsBank call adds `--images data/images` when `IMAGES`
- after the grading calls, if `IMAGES`, run `edge-grader-read-check --label {key}-kaggle-{GPU} --split test_ua --n {READ_CHECK_N} --out results/kaggle`
- assert `not (IMAGES and (RAG or THINKING or TASK != "scientsbank"))`

Check with `git diff --stat` that only the notebook changed, then commit and push.

- [ ] **Step 3: Upload the images, pilot (the owner, Kaggle)**

Zip and upload `data/images/test_ua` and `data/images/test_uq` as a **private** Kaggle dataset ("edge-grader images").

Pilot: `IMAGES = True`, `SPLITS = ["test_ua"]`, `N = 20`, `READ_CHECK_N = 20`. Check:
- `errors` 0
- `unparsed` near 0
- the read check's median CER is low (say ≤ 0.1)
- **open `raw/read-*.jsonl` and read 5 transcriptions against the answers:** if they're unrelated text, the projector isn't in use; check `/tmp/server-*.log` for the mmproj load line

Pilot files aren't committed.

- [ ] **Step 4: Full runs (the owner, Kaggle)**

Set `SPLITS = ["test_ua", "test_uq"]`, `N = 0`, `READ_CHECK_N = 100`; both models. Unzip into `results/kaggle`, commit and push. test_ud only if time allows: render it first with `edge-grader-render --split test_ud`, commit the manifest, and upload.

---

### Task 5: board cost

**Files:** modify `scripts/board_grade.sh`, `tests/test_board_grade.py`, `src/edge_grader/board_check.py` (only `SUMMARY_GLOBS` gains `"img-*.json"`, if B2's stage-prefix lookup is merged; otherwise also add `stage` handling as described in B2 Task 1, Step 4).

**Interfaces:**
- **Env `IMAGES=<images dir>`** (SciEntsBank, base variant only):
  - the server gets `--mmproj $MMPROJ` (`MMPROJ` env, default `$MODEL_DIR/mmproj-<model>.gguf`)
  - the runner gets `--images $IMAGES`
  - folder names `img-<model>-<board>-<split>-<stamp>`

- [ ] **Step 1: Tests first**

```python
def test_images_add_mmproj_and_the_images_dir():
    base = fields(dry("e2b"))
    f = fields(dry("e2b", IMAGES="data/images", MMPROJ="/m/mmproj-e2b.gguf"))
    assert f["server"] == base["server"] + " --mmproj /m/mmproj-e2b.gguf"
    assert f["runner"].endswith(f"--out {f['run_dir']} --images data/images")
    assert re.fullmatch(rf"{REPO}/results/board/img-e2b-jetson-test_ua-{STAMP}", f["run_dir"])
```

Add `"IMAGES"` and `"MMPROJ"` to the variables `dry()` strips. Implement in the same style as B2's `RAG`:
- name prefix `img-`
- `--mmproj` appended to the llama.cpp `server_cmd` branch
- refused with `VARIANT` other than `base`, with `THINKING=on`, or with `RAG`
- `$MMPROJ` added to the "missing file" check
- passed in the inner `exec env`

Then run the tests, lint, commit and push.

- [ ] **Step 2: Put the projector and images on the boards**

```bash
for b in MITLAB-JETSON MITLAB-EDGE; do
  d=$([ $b = MITLAB-JETSON ] && echo research || echo Research)
  rsync -a data/images/test_ua $b:$d/edge-grader/data/images/
  rsync -a results/images $b:$d/edge-grader/results/
done
```

The projector file goes into each board's model directory (`~/research/models` on the Jetson, `/home/mitlab/models` on the Pi). Download it there with `huggingface_hub` from the board's edge-grader venv, or rsync it from the Mac after an `hf_hub_download` there. It's a file download; the model never runs on the Mac.

- [ ] **Step 3: Run** (after phase 0's gate; one job per board)

```bash
ssh MITLAB-JETSON 'cd ~/research/edge-grader && nohup bash -c "for m in e2b e4b; do BENCH=scientsbank IMAGES=data/images MMPROJ=\$HOME/research/models/<mmproj file for \$m> scripts/board_grade.sh \$m test_ua; echo == rc=\$?; sleep 20; done" > ~/c1-jetson.log 2>&1 < /dev/null &'
ssh MITLAB-EDGE   'cd ~/Research/edge-grader && nohup bash -c "for m in e2b e4b; do BENCH=scientsbank IMAGES=data/images MMPROJ=/home/mitlab/models/<mmproj file for \$m> scripts/board_grade.sh \$m test_ua 100; echo == rc=\$?; sleep 20; done" > ~/c1-pi.log 2>&1 < /dev/null &'
```

Fill in the projector file names from Task 1. The Pi runs 100 answers (`n` 100, a seeded sample): image encoding on its CPU is the unknown, and 100 is enough for cost per answer.

Fetch, commit, and run `uv run edge-grader-board-check results/board/img-*`. Each run is paired with the Kaggle `img-*` run of the same split, on the shared ids.

---

### Task 6: C2, the figure-question dataset search

**Files:** create `docs/results/images.md` (section C2); add a row to `docs/research-outline.md`.

- [ ] **Step 1: Search**, with these criteria written at the top of the section **before searching**:
- **(1)** real student answers, free text, not multiple choice
- **(2)** questions that include a figure (diagram, graph, photo) the answer depends on
- **(3)** grade labels or scores per answer
- **(4)** English
- **(5)** a licence or terms allowing research use
- **(6)** roughly 300+ graded answers

Where to look:
- Hugging Face datasets (search terms: short answer grading image, multimodal answer assessment, diagram question student answers)
- papers on multimodal automated short answer grading, and the datasets they release
- the SemEval/BEA shared tasks

Record every candidate in a table, one column per criterion, ticked only after checking the dataset's own card or paper. **Don't list a dataset from memory without opening its page.**

- [ ] **Step 2: Decide with the owner**
- **One qualifies:** write the plan to run it with the C1 image path, the image in the question this time, in a short follow-up plan file.
- **None qualifies:** C2 is dropped. Add an own-decisions row: "C2 dropped: no dataset met the criteria (searched <date>, candidates in docs/results/images.md)".

Commit.

---

### Task 7: results

**Files:** `docs/results/images.md` (C1), `docs/plan.md`, `docs/research-outline.md` (stage C output cell).

- [ ] **Step 1: Paired comparison against the text runs**

```bash
uv run python - <<'EOF'
import json
from pathlib import Path
from edge_grader.board_check import compare, outcomes
K = Path("results/kaggle")
for m in ("e2b", "e4b"):
    for s in ("test_ua", "test_uq"):
        text = max(K.glob(f"baseline-{m}-kaggle-t4-{s}-3way-20261005-*.json"))
        img = max(K.glob(f"img-{m}-kaggle-t4-{s}-3way-*.json"))
        c = compare(outcomes(img), outcomes(text))
        print(f"{m} {s}: text {c['second_accuracy']:.1%} -> image {c['first_accuracy']:.1%}, "
              f"same label {c['same_grade']}/{c['n']}, p {c['mcnemar']['p']}")
    read = max(K.glob(f"read-{m}-kaggle-t4-test_ua-*.json"))
    print(m, "reading check:", json.loads(read.read_text())["metrics"])
EOF
```

- [ ] **Step 2: Split the image-vs-text loss by reading accuracy**

For the 100 read-check ids, compare the image-vs-text grade change between answers read well (CER ≤ 0.1) and answers read poorly. A small table is enough. It says whether lost accuracy comes from reading or from grading.

- [ ] **Step 3: Write `docs/results/images.md` C1**
- setup: font, renderer, projector, prompt line
- the table from Step 1
- the reading check and the split from Step 2
- board cost: J and s per answer, image vs text on the same board
- three plain findings

Then tick `docs/plan.md` and fill in the stage C output cell in `docs/research-outline.md`. Commit and push.
