# Running the baseline on Kaggle (first-time guide)

Kaggle gives free GPU time for Jupyter notebooks. We use it to get **accuracy** numbers; the energy and latency numbers come from the lab boards. Kaggle's interface changes from time to time, so if a button has moved, look for the same words nearby.

## One-time setup

### 1. Kaggle account
1. Sign up at [kaggle.com](https://www.kaggle.com). A Google login is fine.
2. **Verify your phone number:** profile picture → **Settings** → *Phone verification*. Without it, Kaggle won't let a notebook use a GPU or the internet, and this notebook needs both.

### 2. GitHub token, so Kaggle can read the private repo
1. On GitHub: profile picture → **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
2. Fill in:
   - **Name:** `kaggle-edge-grader`
   - **Expiration:** e.g. 90 days
   - **Repository access:** *Only select repositories* → `edge-grader`
   - **Permissions:** Repository permissions → **Contents: Read-only**. Nothing else.
3. Click **Generate** and copy the token. GitHub shows it only once.

The token can only read this one repo. If it ever leaks, delete it on the same page.

## Each time you run

### 3. Import the notebook
1. On Kaggle: **Create** (left sidebar) → **New Notebook**.
2. In the notebook: **File** → **Import Notebook** → upload `notebooks/kaggle_baseline.ipynb` from your clone.

### 4. Session settings (right-hand panel; open it with the arrow at the top right if it's hidden)
- **Accelerator:** **GPU T4 x2** or **GPU P100**. One GPU is enough.
- **Internet:** **On**.

### 5. Add the token as a secret
1. Menu **Add-ons** → **Secrets** → **Add a new secret**.
2. **Label:** `GITHUB_TOKEN` (exactly this); **Value:** the token from step 2.
3. Make sure its **checkbox is ticked**, so this notebook can use it.

The secret is saved on your Kaggle account. Next time you only tick it for the new notebook.

### 6. Run
**Run All** (the ▶▶ button, or menu **Run** → **Run all**). Roughly:

| Cell | Time |
|---|---|
| Build llama-server | 10–15 min |
| Download models | 2–5 min |
| Clone repo, `uv sync`, tests | 1–2 min |
| Grade 100 answers × 2 models | a few min |

Keep the browser tab open. An interactive session stops after about 20 minutes idle, or if the tab is closed for long.

### 7. Check the table in the last cell
- **`unparsed`** should be close to 0. If not, the model is replying in a form the parser doesn't recognise. Send the table and a few `raw` replies from `results/kaggle/raw/*.jsonl`.
- **`errors`** should be 0.
- **`accuracy` / `macro_f1`:** these are the first baseline numbers.

### 8. Save the results
1. Right-hand panel → **Output** (the `/kaggle/working` folder) → `edge-grader/results/kaggle/`. Download the files.
2. Copy them into `results/kaggle/` in your local clone, then commit and push:
   ```bash
   git add results/kaggle && git commit -m "Kaggle baseline: e4b/e2b test_ua n=100" && git push
   ```

## Full runs (all answers) without keeping the tab open

Once the 100-answer run looks right:
1. In the first code cell, set `N = 0` and `SPLITS = ["test_ua", "test_uq", "test_ud"]`.
2. Click **Save Version** (top right) → **Save & Run All (Commit)**.

Kaggle then runs the whole notebook in the background, for up to about 12 hours, and you can close the tab. When it finishes, open the version and download the results from its **Output** tab.

`test_ud` has 4,562 answers, so a full run of both models takes a while.

## Limits to know

- **GPU quota:** about 30 GPU hours a week, shown under **Settings** → quota. A 100-answer run uses very little. Full runs of all splits use more.
- **Nothing persists:** each session starts fresh, which is why the notebook rebuilds llama-server and re-downloads the models each time. The results only survive if you download them, or save a version.
- **Accuracy only:** latency on a Kaggle GPU says nothing about the Pi or Jetson.
