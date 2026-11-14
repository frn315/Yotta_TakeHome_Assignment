"""Ask one question:  python -m rag.ask "what causes ocean tides" """
from __future__ import annotations

import sys

from common.config import Settings
from common.providers import ProviderError, get_chat, get_embedder

from .data import load_corpus
from .pipeline import build_pipeline


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    settings = Settings.from_env()
    try:
        pipeline = build_pipeline(load_corpus(settings).passages, settings, get_chat(settings), get_embedder(settings))
        result = pipeline.answer(" ".join(sys.argv[1:]))
    except ProviderError as exc:
        print(f"Provider error ({exc.kind}): {exc.message}", file=sys.stderr)
        return 1
    print(f"\nAnswer: {result.answer}\n")
    for i, c in enumerate(result.contexts, 1):
        print(f"[{i}] ({c.chunk.passage_id}, grade={c.grade_score:.2f}) {c.chunk.text[:160]}")
    print(f"\nQueries tried: {result.queries_tried}\nTimings (ms): {result.timings_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
