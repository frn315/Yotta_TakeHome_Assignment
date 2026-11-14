"""The RAG pipeline: retrieve -> grade -> (rewrite + retry) -> generate grounded answer.

    question ─► retrieve top-k ─► grade each chunk ─► enough relevant? ─► yes ─► generate
                     ▲                                      │ no
                     └────────── rewrite query ◄────────────┘   (at most RAG_MAX_REWRITES times,
                                                                 then abstain instead of guessing)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from common.config import Settings
from common.events import EventLog
from common.providers import ChatClient, extract_json
from common.text import coverage

from .chunking import Chunk, chunk_corpus
from .retrieval import Hit, build_retriever, to_passages

ABSTAIN = "I don't know based on the provided context."


@dataclass
class GradedChunk:
    chunk: Chunk
    retrieval_score: float
    grade_score: float
    relevant: bool


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[GradedChunk]            # the chunks actually given to the generator
    retrieved_passage_ids: list[str]       # passage-level ranking from the FIRST retrieval
    queries_tried: list[str]
    abstained: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, retriever, chat: ChatClient, settings: Settings, log: EventLog | None = None):
        self.retriever, self.chat, self.s = retriever, chat, settings
        self.log = log or EventLog("rag")

    # ---- grading -------------------------------------------------------------
    def grade(self, question: str, hits: list[Hit]) -> list[GradedChunk]:
        """Threshold grader: share of the question's content terms found in the chunk. It is
        independent of the retriever, so one threshold works for BM25, cosine and RRF scores.
        LLM grader: ONE call judges all k chunks (not k calls); falls back to the threshold."""
        chunks = [self.retriever.chunks[i] for i, _ in hits]
        scores = [coverage(question, c.text) for c in chunks]
        relevant = [s >= self.s.rag_grade_threshold for s in scores]
        if self.s.rag_grader == "llm" and chunks:
            relevant = self._llm_grade(question, chunks) or relevant
        return [GradedChunk(c, h[1], s, r) for c, h, s, r in zip(chunks, hits, scores, relevant)]

    def _llm_grade(self, question: str, chunks: list[Chunk]) -> list[bool] | None:
        numbered = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))
        messages = [
            {"role": "system", "content": "Decide which passages contain information that helps answer the question. "
             'Reply with JSON only: {"relevant": [passage numbers]} (an empty list if none help).'},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n{numbered}"},
        ]
        try:
            picked = {int(n) for n in extract_json(self.chat.complete(messages, json_mode=True))["relevant"]}
            return [i in picked for i in range(1, len(chunks) + 1)]
        except (ValueError, TypeError, KeyError):
            self.log.log("grade_parse_failed", fallback="threshold")
            return None

    # ---- rewrite ---------------------------------------------------------------
    def rewrite(self, question: str, tried: list[str]) -> str:
        messages = [
            {"role": "system", "content": "Rewrite the search query so a retriever finds better passages. Use different "
             "wording: synonyms and the key terms a good answer would contain. Reply with the rewritten query only."},
            {"role": "user", "content": f"Question: {question}\nQueries already tried: {tried}"},
        ]
        return " ".join(self.chat.complete(messages).strip().strip('"').split())[:300]

    # ---- generation ------------------------------------------------------------
    def generate(self, question: str, contexts: list[GradedChunk]) -> str:
        numbered = "\n".join(f"[{i}] {c.chunk.text}" for i, c in enumerate(contexts, 1))
        messages = [
            {"role": "system", "content": "Answer the question using ONLY the numbered context passages. "
             "Be concise (1-3 sentences) and cite the passages you used like [1]. "
             f'If the context does not contain the answer, reply exactly: "{ABSTAIN}"'},
            {"role": "user", "content": f"Context:\n{numbered}\n\nQuestion: {question}"},
        ]
        answer = self.chat.complete(messages).strip()
        self.log.log("answered", contexts=len(contexts), sent=messages[-1]["content"][:4000], reply=answer[:4000])
        return answer

    # ---- orchestration ---------------------------------------------------------
    def answer(self, question: str) -> RAGResult:
        t0 = time.perf_counter()
        timings = {"retrieve": 0.0, "grade": 0.0, "rewrite": 0.0, "generate": 0.0}
        tried = [question]
        best: dict[str, GradedChunk] = {}
        first_ranking: list[str] = []
        query = question
        for attempt in range(self.s.rag_max_rewrites + 1):
            t = time.perf_counter()
            hits = self.retriever.search(query, self.s.rag_top_k)
            timings["retrieve"] += (time.perf_counter() - t) * 1000
            if attempt == 0:
                first_ranking = to_passages(hits, self.retriever.chunks, self.s.rag_top_k)
            t = time.perf_counter()
            graded = self.grade(question, hits)          # always grade against the ORIGINAL question
            timings["grade"] += (time.perf_counter() - t) * 1000
            for g in graded:
                if g.relevant:
                    best.setdefault(g.chunk.chunk_id, g)
            self.log.log("retrieved", attempt=attempt, query=query, hits=len(hits),
                         relevant=sum(g.relevant for g in graded))
            if best or attempt == self.s.rag_max_rewrites:
                break
            t = time.perf_counter()
            query = self.rewrite(question, tried)
            timings["rewrite"] += (time.perf_counter() - t) * 1000
            if not query or query in tried:
                self.log.log("rewrite_stopped", reason="no new query")
                break
            tried.append(query)
            self.log.log("query_rewritten", attempt=attempt + 1, query=query)

        contexts = sorted(best.values(), key=lambda g: -g.grade_score)[: self.s.rag_top_k]
        if not contexts:
            # Stopping condition: no passage passed the grader even after rewriting -> abstain.
            self.log.log("abstained", reason="insufficient context", queries=tried)
            answer, abstained = ABSTAIN, True
        else:
            t = time.perf_counter()
            answer = self.generate(question, contexts)
            timings["generate"] = (time.perf_counter() - t) * 1000
            abstained = ABSTAIN.lower()[:12] in answer.lower()
        timings["total"] = (time.perf_counter() - t0) * 1000
        return RAGResult(question, answer, contexts, first_ranking, tried, abstained,
                         {k: round(v, 2) for k, v in timings.items()})


def build_pipeline(passages: dict[str, str], settings: Settings, chat: ChatClient, embedder) -> RAGPipeline:
    """Wire the configured chunker and retriever into a pipeline (used by ask.py and evaluate.py)."""
    chunks = chunk_corpus(passages, settings.rag_chunker, settings.rag_chunk_words, settings.rag_chunk_overlap)
    retriever = build_retriever(settings.rag_retriever, chunks, embedder)
    return RAGPipeline(retriever, chat, settings, EventLog("rag", settings.log_dir))
