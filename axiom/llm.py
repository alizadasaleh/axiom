"""Local LLM client via Ollama (OD14) -- no API key, no per-call cost, runs on
this machine. Shared by keyword canonicalization (T3.2), reading-list
summaries (T7.4), and (later) the hypothesis pipeline (PBI 5).

Plain httpx against Ollama's REST API (http://localhost:11434) -- httpx is
already a pinned dependency, so this adds no new package; the `ollama` PyPI
client is skipped to keep the dependency list minimal, per this repo's
existing discipline (see axiom/sparse.py, axiom/graph.py for the same stance).
"""
from __future__ import annotations

import json

import httpx

from axiom import config


class OllamaError(RuntimeError):
    """Ollama unreachable, or returned something we couldn't use."""


def chat(
    prompt: str,
    *,
    system: str | None = None,
    model: str = config.OLLAMA_MODEL,
    json_mode: bool = False,
    temperature: float = 0.2,
    timeout: float = config.OLLAMA_TIMEOUT,
) -> str:
    """One-shot chat completion. Returns the assistant's text content.

    `json_mode=True` sets Ollama's `format: "json"` (constrained decoding);
    callers should still parse defensively -- small local models occasionally
    wrap valid JSON in prose or code fences despite the constraint.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        resp = httpx.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Ollama unreachable at {config.OLLAMA_HOST} -- is `ollama serve` running?"
        ) from exc
    return resp.json().get("message", {}).get("content", "")


def chat_json(
    prompt: str, *, system: str | None = None, model: str = config.OLLAMA_MODEL,
    temperature: float = 0.2, timeout: float = config.OLLAMA_TIMEOUT,
) -> dict:
    """chat() + JSON parse, with one fallback extraction if the model wrapped it."""
    text = chat(prompt, system=system, model=model, json_mode=True,
                temperature=temperature, timeout=timeout)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise OllamaError(f"Ollama did not return valid JSON: {text[:200]!r}")
