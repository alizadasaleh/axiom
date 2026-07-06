"""Paper summarization via local Ollama (OD14). Produces a 3-bullet summary of
a paper's abstract, grounded only in the given text -- used by the Reading
List (Task 7.4) and, later, the hypothesis pipeline's Ingest_Summarizer node
(PBI 5).
"""
from __future__ import annotations

from dataclasses import dataclass

from axiom import config, llm

_SYSTEM = (
    "You summarize a single research paper abstract into exactly 3 bullet "
    "points. Every bullet must be a claim directly supported by the abstract "
    "text given -- never add outside knowledge, invented numbers, or claims "
    "not present in the text. Keep each bullet under 25 words."
)


@dataclass
class PaperSummary:
    paper_id: str
    bullets: list[str]


def _prompt(title: str, abstract: str) -> str:
    return (
        f"Title: {title}\n\nAbstract: {abstract}\n\n"
        'Return JSON: {"bullets": ["...", "...", "..."]} -- exactly 3 bullets.'
    )


def summarize_paper(paper_id: str, title: str | None, abstract: str | None,
                     *, model: str = config.OLLAMA_MODEL) -> PaperSummary:
    if not abstract:
        raise ValueError(f"paper {paper_id} has no abstract to summarize")
    data = llm.chat_json(_prompt(title or "", abstract), system=_SYSTEM, model=model)
    bullets = [b for b in data.get("bullets", []) if b][:3]
    if not bullets:
        raise llm.OllamaError(f"model returned no usable bullets for {paper_id}")
    return PaperSummary(paper_id=paper_id, bullets=bullets)
