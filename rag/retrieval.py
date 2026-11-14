"""Dense, keyword (BM25) and hybrid (reciprocal-rank fusion) retrievers over chunks."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from common.providers import EmbeddingClient
from common.text import content_terms

from .chunking import Chunk

Hit = tuple[int, float]  # (chunk index, score)


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(content_terms(c.text)) for c in chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(chunks) else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(self.docs):
            for t, tf in d.items():
                self.postings.setdefault(t, []).append((i, tf))

    def search(self, query: str, k: int) -> list[Hit]:
        scores: dict[int, float] = {}
        for term in set(content_terms(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1.0))
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: EmbeddingClient):
        self.chunks, self.embedder = chunks, embedder
        self.matrix = embedder.embed([c.text for c in chunks], input_type="passage")  # L2-normalised rows

    def search(self, query: str, k: int) -> list[Hit]:
        q = self.embedder.embed([query], input_type="query")[0]
        sims = self.matrix @ q                       # cosine similarity
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        return sorted(((int(i), float(sims[i])) for i in top), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """Reciprocal-rank fusion: score = sum over retrievers of 1 / (rrf_k + rank).
    Rank-based, so BM25 scores and cosine similarities need no calibration."""

    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, rrf_k: int = 60, depth: int = 30):
        self.dense, self.bm25, self.rrf_k, self.depth = dense, bm25, rrf_k, depth
        self.chunks = dense.chunks

    def search(self, query: str, k: int) -> list[Hit]:
        fused: dict[int, float] = {}
        for retriever in (self.dense, self.bm25):
            for rank, (idx, _) in enumerate(retriever.search(query, max(k, self.depth)), start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def build_retriever(kind: str, chunks: list[Chunk], embedder: EmbeddingClient):
    if kind == "bm25":
        return BM25Retriever(chunks)
    if kind == "dense":
        return DenseRetriever(chunks, embedder)
    if kind == "hybrid":
        return HybridRetriever(DenseRetriever(chunks, embedder), BM25Retriever(chunks))
    raise ValueError(f"unknown retriever '{kind}' (use dense, bm25 or hybrid)")


def to_passages(hits: list[Hit], chunks: list[Chunk], k: int) -> list[str]:
    """Collapse chunk hits to distinct parent passage ids, keeping rank order."""
    out: list[str] = []
    for idx, _ in hits:
        pid = chunks[idx].passage_id
        if pid not in out:
            out.append(pid)
        if len(out) == k:
            break
    return out
