"""Small text helpers shared by retrieval, grading and metrics."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

STOPWORDS = frozenset("""
a an the and or but if then of in on at to for from by with without about into over under as is are was were be been being
do does did doing have has had having it its this that these those i you he she we they them his her their our your my me
what which who whom whose when where why how can could should would will shall may might must not no so than too very
there here also just only any some such each other more most much many few own same both up down out off again once
""".split())


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def stem(token: str) -> str:
    """Very light stemmer: enough to match plurals and common verb endings."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_terms(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def coverage(query: str, passage: str) -> float:
    """Fraction of the query's content terms that appear in the passage (0..1)."""
    q = set(content_terms(query))
    if not q:
        return 0.0
    return len(q & set(content_terms(passage))) / len(q)
