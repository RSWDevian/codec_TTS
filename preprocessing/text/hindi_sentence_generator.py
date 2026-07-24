"""Generate Hindi sentences for TTS data collection using a local Ollama model.

Used by scripts/record_studio.py to produce batches of sentences that a
speaker then reads aloud and records, building data/raw/hindi/*.wav + *.txt
pairs for scripts/tokenize_dataset.py.
"""

from __future__ import annotations

import re

import requests

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3:latest"

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LEADING_NUMBER_RE = re.compile(r"^\s*\d+[.)\-]\s*")
LIST_MARKER_RE = re.compile(r"(?:^|\s)\d{1,3}[.)]\s+")
PAREN_RE = re.compile(r"\([^)]*\)")
LATIN_RE = re.compile(r"[A-Za-z]")
TRAILING_GLOSS_RE = re.compile(r"\s+[-–—]\s+.*$")

PROMPT_TEMPLATE = """You are helping build a Hindi text-to-speech dataset.
Write {n} short, natural, everyday Hindi sentences in Devanagari script.

Rules:
- Each sentence must be plain conversational Hindi, no English words.
- Vary sentence length (roughly 5 to 15 words) and topic.
- Write exactly one sentence per line.
- Number each line like "1. <sentence>".
- Do not repeat a sentence.
- Output only the numbered list, no heading, translation, or explanation.
{avoid_clause}"""


def _build_prompt(n: int, avoid: list[str] | None) -> str:
    avoid_clause = ""
    if avoid:
        sample = avoid[-20:]
        bullets = "\n".join(f"- {s}" for s in sample)
        avoid_clause = f"Do not reuse any of these already-used sentences:\n{bullets}\n"
    return PROMPT_TEMPLATE.format(n=n, avoid_clause=avoid_clause)


def _clean_sentence(line: str) -> str | None:
    """Strips numbering/parenthetical transliterations/English glosses; rejects Latin leftovers."""
    line = LEADING_NUMBER_RE.sub("", line)
    line = PAREN_RE.sub("", line)
    line = TRAILING_GLOSS_RE.sub("", line)
    line = re.sub(r"\s+", " ", line).strip()
    if not line or not DEVANAGARI_RE.search(line) or LATIN_RE.search(line):
        return None
    return line


def _split_response(text: str) -> list[str]:
    """Splits on numbered-list markers first (models often cram the whole list onto one
    line); falls back to newlines if no markers were found."""
    parts = [p for p in LIST_MARKER_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        parts = [line for line in text.splitlines() if line.strip()]
    return parts


def _parse_sentences(text: str) -> list[str]:
    sentences = []
    for chunk in _split_response(text):
        cleaned = _clean_sentence(chunk)
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _call_ollama(prompt: str, model: str, host: str, timeout: float) -> str:
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host} (model={model}). "
            f"Is `ollama serve` running and is the model pulled? Original error: {e}"
        ) from e
    return resp.json().get("response", "")


def generate_batch(
    n: int = 20,
    model: str = DEFAULT_MODEL,
    host: str = OLLAMA_HOST,
    timeout: float = 120.0,
    max_attempts: int = 3,
) -> list[str]:
    """Generate up to `n` unique Hindi sentences, retrying if the model under-delivers."""
    collected: list[str] = []
    seen: set[str] = set()

    for _ in range(max_attempts):
        remaining = n - len(collected)
        if remaining <= 0:
            break
        prompt = _build_prompt(remaining, collected)
        raw = _call_ollama(prompt, model, host, timeout)
        for sentence in _parse_sentences(raw):
            if sentence not in seen:
                seen.add(sentence)
                collected.append(sentence)
                if len(collected) >= n:
                    break

    return collected[:n]
