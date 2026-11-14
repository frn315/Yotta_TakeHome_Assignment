"""Structured event log shared by both workstreams.

Every agent decision, action, validation failure and edit operation is recorded
as one JSON object. Events are (a) kept in memory so the API can return them to
the UI, (b) appended to logs/events.jsonl, and (c) mirrored to stdlib logging.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


class EventLog:
    def __init__(self, name: str, log_dir: Path | None = None, context: dict | None = None):
        self.name = name
        self.events: list[dict[str, Any]] = []
        self.context = context or {}
        self._logger = logging.getLogger(name)
        self._path: Path | None = None
        if log_dir is not None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                self._path = log_dir / "events.jsonl"
            except OSError:
                self._path = None  # logging must never break the request

    def log(self, kind: str, **data: Any) -> dict:
        event = {"ts": round(time.time(), 3), "source": self.name, "kind": kind, **self.context, **data}
        self.events.append(event)
        level = logging.WARNING if kind.endswith(("failed", "error")) else logging.INFO
        self._logger.log(level, "%s %s", kind, json.dumps(data, default=str)[:500])
        if self._path is not None:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")
            except OSError:
                pass
        return event
