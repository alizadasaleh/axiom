# data_prep — Corpus building & cleaning

Exploratory and productionized steps that turn the raw `paperlists/` archives
into a clean, model-ready corpus.

| File               | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `flow.ipynb`       | Exploratory notebook: crawl `paperlists/`, profile text artifacts, derive the cleaning rules. |
| `clean_dataset.py` | Productionized cleaner encoding those rules (dedupe, drop admin rows, normalize text). |

## Usage

Run from the **repo root** (paths default to root-level files):

```bash
python data_prep/clean_dataset.py --input FULL_DATASET.jsonl --output cleaned_dataset.jsonl
```

## Data files (not committed)

`FULL_DATASET.jsonl` and `cleaned_dataset.jsonl` (~78 MB each) live at the repo
root and are **git-ignored** — regenerate them with the notebook / cleaner
rather than committing them.
