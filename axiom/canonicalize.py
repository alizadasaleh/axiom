"""LLM keyword canonicalization (Task 3.2, OD14): merge concept-label synonyms
(e.g. "LoRA" / "Low-Rank Adaptation") into one canonical form using a local
Ollama model -- no API key, no cost.

Batches distinct concept labels (default 50/batch, per the backlog spec),
asks the model to group true synonyms and name each group's canonical label,
and persists the mapping to `concept_canonical` (db/schema.sql) so it's
computed once per corpus, not on every velocity query. Idempotent: re-running
overwrites prior `source='auto'` rows but never touches `source='manual'`
rows (the manual-override path the backlog's acceptance criteria call for).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from axiom import config, db, llm

_SYSTEM = (
    "You are grouping near-duplicate machine learning research concept labels "
    "into canonical forms. Group ONLY true synonyms or abbreviation/full-name "
    "pairs (e.g. 'LoRA' and 'Low-Rank Adaptation'). Never merge merely-related "
    "but distinct concepts (e.g. 'Machine translation' and 'Low-resource NLP' "
    "must stay separate groups). Every input label must appear in exactly one "
    "group, even if that group has only one member."
)


def _prompt(labels: list[str]) -> str:
    joined = "\n".join(f"- {label}" for label in labels)
    return (
        f"Concept labels:\n{joined}\n\n"
        'Return JSON: {"groups": [{"canonical": "<label>", "members": '
        '["<label>", ...]}, ...]}. `canonical` must be one of `members` '
        "(pick the clearer/more complete form)."
    )


@dataclass
class CanonicalizationResult:
    mapping: dict[str, str]      # concept -> canonical, one entry per input label
    groups: list[list[str]]      # non-trivial groups only (size > 1), for review


def canonicalize_batch(labels: list[str], *, model: str = config.OLLAMA_MODEL) -> CanonicalizationResult:
    data = llm.chat_json(_prompt(labels), system=_SYSTEM, model=model)
    mapping: dict[str, str] = {}
    groups: list[list[str]] = []
    label_set = set(labels)
    for g in data.get("groups", []):
        canonical = g.get("canonical")
        members = [m for m in g.get("members", []) if m in label_set]
        if not canonical or not members:
            continue
        for m in members:
            mapping.setdefault(m, canonical)
        if len(members) > 1:
            groups.append(members)
    # Any label the model dropped or malformed maps to itself (safe fallback).
    for label in labels:
        mapping.setdefault(label, label)
    return CanonicalizationResult(mapping=mapping, groups=groups)


def run(conn: sqlite3.Connection, *, batch_size: int = 50,
        model: str = config.OLLAMA_MODEL, log=print) -> int:
    """Canonicalize every distinct concept in the corpus; persist to SQLite.

    Skips concepts already manually overridden. Returns the number of
    (concept -> canonical) rows written.
    """
    labels = [r["concept"] for r in conn.execute(
        "SELECT DISTINCT concept FROM concepts ORDER BY concept"
    ).fetchall()]
    manual = db.manual_canonical_concepts(conn)
    todo = [label for label in labels if label not in manual]

    written = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        result = canonicalize_batch(batch, model=model)
        for concept, canonical in result.mapping.items():
            db.upsert_canonical(conn, concept, canonical, source="auto")
            written += 1
        if result.groups:
            merged = sum(len(g) for g in result.groups)
            log(f"[canonicalize] batch {start // batch_size + 1}: "
                f"merged {merged} labels into {len(result.groups)} groups")
    conn.commit()
    log(f"[canonicalize] {written} concepts mapped "
        f"({len(manual)} manual overrides preserved)")
    return written
