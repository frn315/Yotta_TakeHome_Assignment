"""Versioned website state. Every successful generate/edit/undo appends a version.

In-memory by design (single-process demo). The interface is small so it can be
swapped for SQLite/Redis without touching the agents or the API.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid


class SiteNotFound(KeyError):
    pass


class StaleVersion(Exception):
    pass


def _entry(version: int, site: dict, source: str, instruction: str, ops: list[dict] | None = None, **extra) -> dict:
    return {"version": version, "site": copy.deepcopy(site), "source": source, "instruction": instruction,
            "ops": ops or [], "ts": time.time(), **extra}


class SiteStore:
    def __init__(self) -> None:
        self._sites: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def _versions(self, site_id: str, base_version: int | None = None) -> list[dict]:
        """The site's history; refuses a caller whose `base_version` is no longer the latest."""
        try:
            versions = self._sites[site_id]
        except KeyError:
            raise SiteNotFound(site_id) from None
        latest = versions[-1]["version"]
        if base_version not in (None, latest):
            raise StaleVersion(f"Preview is at version {base_version} but the site is at version {latest}. "
                               "Refresh and try again.")
        return versions

    def create(self, site: dict, brief: str) -> tuple[str, dict]:
        site_id = uuid.uuid4().hex[:12]
        entry = _entry(1, site, "generate", brief)
        with self._lock:
            self._sites[site_id] = [entry]
        return site_id, copy.deepcopy(entry)

    def current(self, site_id: str, base_version: int | None = None) -> dict:
        with self._lock:
            return copy.deepcopy(self._versions(site_id, base_version)[-1])

    def commit(self, site_id: str, site: dict, *, instruction: str, ops: list[dict], base_version: int) -> dict:
        """Append an edit. `base_version` is the version the edit was computed from: if the site has
        moved on since, the edit is refused instead of silently overwriting the newer version."""
        with self._lock:
            versions = self._versions(site_id, base_version)
            versions.append(_entry(base_version + 1, site, "edit", instruction, ops))
            return copy.deepcopy(versions[-1])

    def undo(self, site_id: str) -> dict | None:
        """Revert to the previous state by appending it as a new version (history is kept)."""
        with self._lock:
            versions = self._versions(site_id)
            idx = self._undo_target(versions)
            if idx is None:
                return None
            target = versions[idx]
            versions.append(_entry(versions[-1]["version"] + 1, target["site"], "undo",
                                   f"undo to v{target['version']}", restored_from=target["version"]))
            return copy.deepcopy(versions[-1])

    @staticmethod
    def _undo_target(versions: list[dict]) -> int | None:
        current = len(versions) - 1
        # If the current version is itself an undo, continue from what it restored.
        while versions[current]["source"] == "undo":
            current = versions[current]["restored_from"] - 1
        return current - 1 if current >= 1 else None

    def history(self, site_id: str) -> list[dict]:
        with self._lock:
            return [{k: v for k, v in e.items() if k != "site"} for e in self._versions(site_id)]
