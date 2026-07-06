#!/usr/bin/env python3
"""Clean the ACL/NLP JSONL corpus into a model-ready dataset.

This script encodes the cleaning rules discovered in `flow.ipynb`:
- keep only real paper records
- drop administrative / structural entries
- remove duplicate ids
- normalize whitespace and common text artifacts
- preserve Unicode content while removing obvious control noise

Input defaults to `FULL_DATASET.jsonl`.
Output defaults to `cleaned_dataset.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


ADMIN_BLACKLIST = (
    "preface",
    "table of contents",
    "acknowledgment",
    "organisation",
    "organization",
    "committee",
    "session",
    "index",
    "introduction by",
    "volume",
    "proceedings of",
    "reviewer acknowledgments",
)

LATEX_COMMANDS = (
    (r"\\textquotedblleft", '"'),
    (r"\\textquotedblright", '"'),
    (r"\\infty", "infinity"),
    (r"\\%", "%"),
    (r"\\&", "&"),
    (r"\\_", "_"),
)

WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
LATEX_NOISE_RE = re.compile(r"[{}]|\\[a-zA-Z]+")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    for pattern, replacement in LATEX_COMMANDS:
        text = re.sub(pattern, replacement, text)
    text = LATEX_NOISE_RE.sub(" ", text)
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_valid_title(title: str) -> bool:
    if not title:
        return False
    lowered = title.lower()
    if any(term in lowered for term in ADMIN_BLACKLIST):
        return False
    if len(title.split()) < 3:
        return False
    return True


def clean_record(record: Dict[str, Any]) -> Dict[str, Any] | None:
    title = normalize_text(record.get("title"))
    abstract = normalize_text(record.get("abstract"))

    if not is_valid_title(title):
        return None
    if not abstract:
        return None

    cleaned = dict(record)
    cleaned["title"] = title
    cleaned["abstract"] = abstract

    if "venue" in cleaned and cleaned["venue"] is not None:
        cleaned["venue"] = normalize_text(cleaned["venue"]).upper()
    if "year" in cleaned:
        year = cleaned["year"]
        try:
            cleaned["year"] = int(year)
        except Exception:
            cleaned["year"] = year

    return cleaned


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="FULL_DATASET.jsonl",
        type=Path,
        help="Source JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="cleaned_dataset.jsonl",
        type=Path,
        help="Cleaned JSONL file.",
    )
    args = parser.parse_args()

    seen_ids = set()
    stats = Counter()

    with args.output.open("w", encoding="utf-8") as out:
        for record in load_jsonl(args.input):
            stats["read"] += 1
            paper_id = str(record.get("id", "")).strip()
            if not paper_id:
                stats["dropped_missing_id"] += 1
                continue
            if paper_id in seen_ids:
                stats["dropped_duplicate_id"] += 1
                continue

            cleaned = clean_record(record)
            if cleaned is None:
                stats["dropped_invalid_record"] += 1
                continue

            seen_ids.add(paper_id)
            stats["kept"] += 1

            # Preserve the original payload but trim obvious text artifacts in title/abstract.
            out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    print("Cleaning complete.")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Read:   {stats['read']}")
    print(f"Kept:   {stats['kept']}")
    print(f"Dropped duplicate ids: {stats['dropped_duplicate_id']}")
    print(f"Dropped missing ids:   {stats['dropped_missing_id']}")
    print(f"Dropped invalid rows:   {stats['dropped_invalid_record']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
