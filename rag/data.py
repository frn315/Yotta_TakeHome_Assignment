"""Load a small, reproducible MS MARCO subset (with a local fallback) and clean it.

Primary source: Hugging Face `microsoft/ms_marco`, config `v1.1`, validation split.
We take the first N queries (streaming order is deterministic) that have at least
one passage marked `is_selected=1`. Those selected passages are the known-relevant
labels (qrels); the other ~9 passages per query act as hard distractors.

If the download fails for any reason (no network, `datasets` not installed) we
fall back to data/sample_passages.json, which has the same shape.
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from common.config import ROOT, Settings

log = logging.getLogger("rag.data")
SAMPLE_PATH = ROOT / "data" / "sample_passages.json"
CACHE_DIR = ROOT / "data" / "cache"
MIN_WORDS = 5

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")


@dataclass
class Query:
    id: str
    text: str
    relevant_ids: list[str]
    answers: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    source: str                       # "ms_marco_v1.1" or "local-sample"
    passages: dict[str, str]          # passage_id -> cleaned text
    queries: list[Query]


def clean_text(text: str) -> str:
    """Unescape HTML entities, drop control characters, collapse whitespace."""
    text = html.unescape(text or "")
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _pid(text: str) -> str:
    return "m" + hashlib.md5(text.lower().encode()).hexdigest()[:10]


def _from_rows(rows: list[dict], source: str) -> Corpus:
    """rows: [{query_id, query, answers, passages: [{text, is_selected}]}] -> Corpus (cleaned, de-duplicated)."""
    passages: dict[str, str] = {}
    queries: list[Query] = []
    dropped = 0
    for row in rows:
        relevant: list[str] = []
        for p in row["passages"]:
            text = clean_text(p["text"])
            if len(text.split()) < MIN_WORDS:
                dropped += 1
                continue
            pid = p.get("id") or _pid(text)
            passages.setdefault(pid, text)      # identical text -> same id -> de-duplicated
            if p.get("is_selected"):
                relevant.append(pid)
        if relevant:
            answers = [clean_text(a) for a in row.get("answers", []) if a and "No Answer Present" not in a]
            queries.append(Query(str(row["query_id"]), clean_text(row["query"]), sorted(set(relevant)), answers))
    log.info("corpus source=%s queries=%d passages=%d dropped_short=%d", source, len(queries), len(passages), dropped)
    return Corpus(source, passages, queries)


def load_local_sample(path: Path = SAMPLE_PATH) -> Corpus:
    raw = json.loads(path.read_text(encoding="utf-8"))
    text_of = {p["id"]: p["text"] for p in raw["passages"]}
    by_query = [{"query_id": q["id"], "query": q["query"], "answers": q.get("answers", []),
                 "passages": [{"id": pid, "text": text_of[pid], "is_selected": 1}
                              for pid in set(q["relevant_passage_ids"])]}
                for q in raw["queries"]]
    corpus = _from_rows(by_query, raw.get("source", "local-sample"))
    for p in raw["passages"]:                 # add the distractors
        text = clean_text(p["text"])
        if len(text.split()) >= MIN_WORDS:
            corpus.passages.setdefault(p["id"], text)
    return corpus


def load_ms_marco(num_queries: int) -> Corpus:
    cache = CACHE_DIR / f"msmarco_v1.1_validation_first{num_queries}.json"
    if cache.exists():
        return _from_rows(json.loads(cache.read_text(encoding="utf-8")), "ms_marco_v1.1")
    from datasets import load_dataset  # imported lazily: optional dependency

    stream = load_dataset("microsoft/ms_marco", "v1.1", split="validation", streaming=True)
    rows: list[dict] = []
    for row in stream:
        selected = row["passages"]["is_selected"]
        if not any(selected):
            continue
        rows.append({
            "query_id": row["query_id"], "query": row["query"], "answers": list(row.get("answers") or []),
            "passages": [{"text": t, "is_selected": int(s)}
                         for t, s in zip(row["passages"]["passage_text"], selected)],
        })
        if len(rows) >= num_queries:
            break
    if not rows:
        raise RuntimeError("MS MARCO stream returned no usable rows.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows), encoding="utf-8")   # freeze the subset for reproducibility
    return _from_rows(rows, "ms_marco_v1.1")


def load_corpus(settings: Settings) -> Corpus:
    if not settings.rag_force_local:
        try:
            return load_ms_marco(settings.rag_num_queries)
        except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
            log.warning("MS MARCO load failed (%s: %s); using local sample %s",
                        type(exc).__name__, str(exc)[:120], SAMPLE_PATH.name)
    return load_local_sample()
