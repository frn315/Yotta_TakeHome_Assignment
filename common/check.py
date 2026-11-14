"""Provider smoke test:  python -m common.check

Sends one tiny request per capability to the configured endpoints and reports
what works, so a new key / model / gateway can be verified in seconds before
running the app or the evaluations.
"""
from __future__ import annotations

import time

from .config import Settings
from .providers import ProviderError, extract_json, get_chat, get_embedder


def _run(label: str, fn) -> bool:
    started = time.perf_counter()
    try:
        detail = fn()
        print(f"  ok    {label:<22} {time.perf_counter() - started:5.1f}s  {detail}")
        return True
    except (ProviderError, ValueError) as exc:
        kind = getattr(exc, "kind", "invalid_json")
        print(f"  FAIL  {label:<22} {time.perf_counter() - started:5.1f}s  {kind}: {str(exc)[:160]}")
        return False


def main() -> int:
    settings = Settings.from_env()
    print("LLM       ", settings.llm.public())
    print("Embeddings", settings.embedding.public())
    ok = True
    try:
        chat = get_chat(settings)
        say = [{"role": "user", "content": "Reply with the single word: pong"}]
        as_json = [{"role": "user", "content": 'Reply with JSON only: {"status": "ok"}'}]
        ok &= _run("chat", lambda: repr(chat.complete(say)[:40]))
        ok &= _run("chat, JSON output", lambda: extract_json(chat.complete(as_json, json_mode=True)))
    except ProviderError as exc:
        ok = False
        print(f"  FAIL  chat client: {exc.kind}: {exc.message}")
    try:
        embedder = get_embedder(settings)
        ok &= _run("embeddings", lambda: f"shape={embedder.embed(['hello world'], 'query').shape}")
    except ProviderError as exc:
        ok = False
        print(f"  FAIL  embeddings client: {exc.kind}: {exc.message}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
