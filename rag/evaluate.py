"""Evaluation for Workstream A.

    python -m rag.evaluate                 # full run, writes results/rag_*.{csv,json}
    python -m rag.evaluate --llm-judge     # also ask the LLM to judge faithfulness

Part 1  Retrieval comparison: {fixed, sentence} chunking x {dense, bm25, hybrid}
        -> Recall@1/3/5, MRR@10, mean query latency.
Part 2  End-to-end with the configured retriever/chunker: answer F1 against the
        reference answer, faithfulness (is the answer supported by its context),
        abstention rate, rewrite rate, latency per stage.
Part 3  Failure cases: queries whose relevant passage was not in the top 5.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.config import ROOT, Settings
from common.providers import ProviderError, extract_json, get_chat, get_embedder
from common.text import content_terms, tokenize

from .chunking import chunk_corpus
from .data import Corpus, load_corpus
from .pipeline import build_pipeline
from .retrieval import build_retriever, to_passages

RESULTS = ROOT / "results"
KS = (1, 3, 5)
_CITE = re.compile(r"\[\d+\]")


# ------------------------------------------------------------------ metrics
def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: list[str], k: int = 10) -> float:
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def token_f1(prediction: str, reference: str) -> float:
    p, r = tokenize(_CITE.sub("", prediction)), tokenize(reference)
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def lexical_faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of the answer's content terms that appear in the supplied context.
    1.0 = every content word is supported. Cheap proxy: catches invented facts/entities,
    but not wrong relations between supported words."""
    terms = set(content_terms(_CITE.sub("", answer)))
    if not terms:
        return 0.0
    return len(terms & set(content_terms(" ".join(contexts)))) / len(terms)


def llm_faithfulness(chat, answer: str, contexts: list[str]) -> float | None:
    messages = [
        {"role": "system", "content": "Judge whether every claim in the answer is supported by the context. "
         'Reply with JSON only: {"score": number from 0 (unsupported) to 1 (fully supported)}'},
        {"role": "user", "content": "Context:\n" + "\n".join(contexts) + f"\n\nAnswer: {answer}"},
    ]
    try:
        return max(0.0, min(1.0, float(extract_json(chat.complete(messages, json_mode=True))["score"])))
    except (ValueError, TypeError, KeyError, ProviderError):
        return None


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


# ------------------------------------------------------------------ part 1
def retrieval_grid(corpus: Corpus, settings: Settings, embedder) -> tuple[list[dict], list[dict]]:
    rows, failures = [], []
    for chunker in ("fixed", "sentence"):
        chunks = chunk_corpus(corpus.passages, chunker, settings.rag_chunk_words, settings.rag_chunk_overlap)
        for kind in ("dense", "bm25", "hybrid"):
            t = time.perf_counter()
            retriever = build_retriever(kind, chunks, embedder)
            build_ms = (time.perf_counter() - t) * 1000
            recalls = {k: [] for k in KS}
            rrs, latencies = [], []
            for q in corpus.queries:
                t = time.perf_counter()
                hits = retriever.search(q.text, 30)
                latencies.append((time.perf_counter() - t) * 1000)
                ranked = to_passages(hits, chunks, 10)
                for k in KS:
                    recalls[k].append(recall_at_k(ranked, q.relevant_ids, k))
                rrs.append(reciprocal_rank(ranked, q.relevant_ids))
                if recall_at_k(ranked, q.relevant_ids, 5) == 0:
                    failures.append({"chunker": chunker, "retriever": kind, "query_id": q.id, "query": q.text,
                                     "relevant_ids": q.relevant_ids, "top5": ranked[:5],
                                     "top1_text": corpus.passages[ranked[0]][:200] if ranked else ""})
            rows.append({
                "chunker": chunker, "retriever": kind, "num_chunks": len(chunks), "num_queries": len(corpus.queries),
                **{f"recall@{k}": round(statistics.mean(recalls[k]), 4) for k in KS},
                "mrr@10": round(statistics.mean(rrs), 4),
                "mean_query_ms": round(statistics.mean(latencies), 3),
                "index_build_ms": round(build_ms, 1),
            })
    return rows, failures


# ------------------------------------------------------------------ part 2
def end_to_end(corpus: Corpus, settings: Settings, embedder, chat, llm_judge: bool, workers: int) -> tuple[list[dict], dict]:
    pipeline = build_pipeline(corpus.passages, settings, chat, embedder)

    def run(q) -> dict:
        row = {"query_id": q.id, "query": q.text, "answer": "", "reference": q.answers[0] if q.answers else "", "error": ""}
        try:
            result = pipeline.answer(q.text)
        except ProviderError as exc:           # one failed call must not sink the whole run
            return row | {"error": f"{exc.kind}: {exc.message[:120]}"}
        contexts = [c.chunk.text for c in result.contexts]
        f1 = max((token_f1(result.answer, a) for a in q.answers), default=None)
        judge = llm_faithfulness(chat, result.answer, contexts) if llm_judge and not result.abstained else None
        return row | {
            "answer": result.answer, "abstained": result.abstained, "rewrites": len(result.queries_tried) - 1,
            "queries_tried": " | ".join(result.queries_tried),
            "recall@5": recall_at_k(result.retrieved_passage_ids, q.relevant_ids, 5),
            "relevant_in_context": bool({c.chunk.passage_id for c in result.contexts} & set(q.relevant_ids)),
            "answer_f1": "" if f1 is None else round(f1, 4),
            "faithfulness_lexical": "" if result.abstained else round(lexical_faithfulness(result.answer, contexts), 4),
            "faithfulness_llm": "" if judge is None else round(judge, 4),
            **{f"{k}_ms": v for k, v in result.timings_ms.items()},
        }

    # Queries are independent and the time is spent waiting on the provider, so run them concurrently.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        all_rows = list(pool.map(run, corpus.queries))
    rows = [r for r in all_rows if not r["error"]]
    if not rows:
        raise SystemExit(f"Every query failed. First error: {all_rows[0]['error']}")

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] != ""]
        return round(statistics.mean(vals), 4) if vals else None

    totals = [r["total_ms"] for r in rows]
    summary = {
        "num_queries": len(rows), "failed_queries": len(all_rows) - len(rows),
        "answer_f1_mean": mean("answer_f1"),
        "faithfulness_lexical_mean": mean("faithfulness_lexical"),
        "faithfulness_llm_mean": mean("faithfulness_llm"),
        "recall@5": mean("recall@5"),
        "relevant_in_context_rate": round(sum(r["relevant_in_context"] for r in rows) / len(rows), 4),
        "abstention_rate": round(sum(r["abstained"] for r in rows) / len(rows), 4),
        "rewrite_rate": round(sum(r["rewrites"] > 0 for r in rows) / len(rows), 4),
        "latency_ms": {"p50": round(pct(totals, 0.5), 2), "p95": round(pct(totals, 0.95), 2),
                       "mean": round(statistics.mean(totals), 2),
                       **{f"{s}_mean": round(statistics.mean(r[f"{s}_ms"] for r in rows), 2)
                          for s in ("retrieve", "grade", "rewrite", "generate")}},
    }
    return all_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(max(rows, key=len)), restval="")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm-judge", action="store_true", help="also score faithfulness with the LLM")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the end-to-end part (needs no LLM)")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N queries")
    parser.add_argument("--workers", type=int, default=4, help="concurrent queries in the end-to-end part")
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    embedder = get_embedder(settings)
    corpus = load_corpus(settings)
    if args.limit:
        corpus.queries = corpus.queries[: args.limit]

    grid, failures = retrieval_grid(corpus, settings, embedder)
    write_csv(out / "rag_retrieval_metrics.csv", grid)
    e2e_summary = None
    if not args.retrieval_only:
        e2e_rows, e2e_summary = end_to_end(corpus, settings, embedder, get_chat(settings), args.llm_judge, args.workers)
        write_csv(out / "rag_e2e_per_query.csv", e2e_rows)
    (out / "rag_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    summary = {
        "dataset": {"source": corpus.source, "num_queries": len(corpus.queries), "num_passages": len(corpus.passages)},
        "providers": settings.public(),
        "config": {"chunk_words": settings.rag_chunk_words, "chunk_overlap": settings.rag_chunk_overlap,
                   "top_k": settings.rag_top_k, "retriever": settings.rag_retriever, "chunker": settings.rag_chunker,
                   "grader": settings.rag_grader, "grade_threshold": settings.rag_grade_threshold,
                   "max_rewrites": settings.rag_max_rewrites, "workers": args.workers},
        "retrieval": grid,
        "end_to_end": e2e_summary,
        "retrieval_failures": len(failures),
    }
    (out / "rag_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDataset: {corpus.source}  queries={len(corpus.queries)}  passages={len(corpus.passages)}")
    print(f"LLM: {settings.llm.provider}/{settings.llm.model}   Embeddings: {embedder.provider}/{embedder.model}\n")
    print(f"{'chunker':<10}{'retriever':<10}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR@10':>8}{'ms/query':>10}")
    for r in grid:
        print(f"{r['chunker']:<10}{r['retriever']:<10}{r['recall@1']:>7.3f}{r['recall@3']:>7.3f}{r['recall@5']:>7.3f}"
              f"{r['mrr@10']:>8.3f}{r['mean_query_ms']:>10.2f}")
    print("\nEnd-to-end:", json.dumps(e2e_summary, indent=2))
    print(f"\nWrote results to {out}/")


if __name__ == "__main__":
    main()
