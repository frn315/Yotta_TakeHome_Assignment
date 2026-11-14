"""Evaluation for Workstream B.

    python -m builder.evaluate      # writes results/builder_edits.csv and results/builder_summary.json

Runs the scenarios in data/builder_scenarios.json through the workflow with the
configured provider (real model calls) and reports:
  * generation validity and whether the requested sections were produced
  * edit success rate, overall and per edit type (content / style / structure)
  * preservation rate: after each edit, do ALL earlier edits in the session still hold?
  * isolation: for content/style edits, are untouched sections byte-identical?
  * clean rejection of invalid edits (error returned, state unchanged)
  * repair attempts and latency
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.config import ROOT, Settings
from common.events import EventLog
from common.providers import ProviderError, get_chat

from .agents import Workflow, WorkflowError

SCENARIOS = ROOT / "data" / "builder_scenarios.json"
RESULTS = ROOT / "results"


def _first(site: dict, stype: str) -> dict | None:
    return next((s for s in site["sections"] if s["type"] == stype), None)


def _get(props: dict, path: str):
    node = props
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def check(expect: dict, before: dict, after: dict) -> bool:
    """Does `after` satisfy the expectation? (`before` is the state prior to that edit.)"""
    kind = expect["kind"]
    types = [s["type"] for s in after["sections"]]
    try:
        if kind == "content":
            actual = _get(_first(after, expect["section_type"])["props"], expect["path"])
            return str(actual).strip().lower() == expect["equals"].lower()
        if kind == "style":
            return _first(after, expect["section_type"])["style"][expect["property"]] == expect["equals"]
        if kind == "style_changed":
            prop, st = expect["property"], expect["section_type"]
            return _first(after, st)["style"][prop] != _first(before, st)["style"][prop]
        if kind == "theme_changed":
            return after["theme"][expect["property"]] != before["theme"][expect["property"]]
        if kind == "has_section":
            return expect["section_type"] in types
        if kind == "no_section":
            return expect["section_type"] not in types
        if kind == "order":
            return types.index(expect["first"]) < types.index(expect["then"])
        if kind == "last":
            return types[-1] == expect["section_type"]
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return False


def untouched_identical(before: dict, after: dict, changed_ids: set[str]) -> bool:
    b = {s["id"]: s for s in before["sections"]}
    a = {s["id"]: s for s in after["sections"]}
    return all(a.get(sid) == sec for sid, sec in b.items() if sid not in changed_ids)


def run_scenario(sc: dict, chat, settings: Settings) -> tuple[dict, list[dict]]:
    wf = Workflow(chat, EventLog("builder.eval", settings.log_dir, {"scenario": sc["name"]}), settings.builder_max_repairs)
    started = time.perf_counter()
    try:
        site = wf.generate(sc["brief"])
    except (WorkflowError, ProviderError) as exc:
        return {"scenario": sc["name"], "valid": False, "error": str(exc)[:200]}, []
    types = [s["type"] for s in site["sections"]]
    generation = {
        "scenario": sc["name"], "valid": True, "sections": types,
        "requested_sections_present": all(x in types for x in sc["expect_sections"]),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "repairs": sum(e["kind"] == "validation_failed" for e in wf.log.events),
    }
    rows: list[dict] = []
    held: list[tuple[dict, dict]] = []      # (expectation, state before that edit) for edits that succeeded
    for step, edit in enumerate(sc["edits"], start=1):
        before, expect = site, edit["expect"]
        row = {"scenario": sc["name"], "step": step, "type": edit["type"], "instruction": edit["instruction"],
               "applied": False, "correct": False, "earlier_edits_preserved": "", "untouched_sections_identical": "",
               "attempts": "", "latency_ms": "", "ops": "", "error": ""}
        started = time.perf_counter()
        try:
            result = wf.edit(site, edit["instruction"])
        except WorkflowError as exc:
            row["error"] = exc.reason[:200]
            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            row["correct"] = expect["kind"] == "rejected"   # state is untouched by construction
        except ProviderError as exc:
            row["error"] = f"provider {exc.kind}: {exc.message[:140]}"
        else:
            site = result.site
            row.update(applied=True, attempts=result.attempts, latency_ms=result.latency_ms,
                       ops=json.dumps(result.ops, separators=(",", ":")))
            row["correct"] = expect["kind"] != "rejected" and check(expect, before, site)
            if edit["type"] in ("content", "style") and all("section_id" in c for c in result.changes):
                row["untouched_sections_identical"] = untouched_identical(before, site, {c["section_id"] for c in result.changes})
            earlier = list(held)
            if row["correct"]:
                held.append((expect, before))
            if earlier:
                row["earlier_edits_preserved"] = all(check(e, b, site) for e, b in earlier)
        rows.append(row)
    return generation, rows


def main() -> None:
    settings = Settings.from_env()
    chat = get_chat(settings)
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]
    # Scenarios are independent sessions; run them concurrently (the time is provider latency).
    with ThreadPoolExecutor(max_workers=len(scenarios)) as pool:
        outcomes = list(pool.map(lambda sc: run_scenario(sc, chat, settings), scenarios))
    generations = [g for g, _ in outcomes]
    rows = [r for _, rs in outcomes for r in rs]
    if not rows:
        raise SystemExit(f"No scenario produced a site: {generations}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "builder_edits.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def rate(items: list, key=lambda r: r) -> float | None:
        return round(sum(bool(key(i)) for i in items) / len(items), 4) if items else None

    valid_edits = [r for r in rows if r["type"] != "invalid"]
    invalid_edits = [r for r in rows if r["type"] == "invalid"]
    preserved = [r["earlier_edits_preserved"] for r in rows if r["earlier_edits_preserved"] != ""]
    isolated = [r["untouched_sections_identical"] for r in rows if r["untouched_sections_identical"] != ""]
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] != ""]
    summary = {
        "providers": settings.public(),
        "generation": {"scenarios": len(scenarios), "valid_rate": rate(generations, lambda g: g["valid"]),
                       "requested_sections_rate": rate(generations, lambda g: g.get("requested_sections_present")),
                       "details": generations},
        "edits": {
            "total": len(rows),
            "success_rate_valid_requests": rate(valid_edits, lambda r: r["correct"]),
            "by_type": {t: rate([r for r in valid_edits if r["type"] == t], lambda r: r["correct"])
                        for t in ("content", "style", "structure")},
            "invalid_requests_rejected_cleanly": rate(invalid_edits, lambda r: r["correct"]),
            "earlier_edits_preserved_rate": rate(preserved),
            "untouched_sections_identical_rate": rate(isolated),
            "mean_attempts": round(statistics.mean(r["attempts"] for r in rows if r["attempts"] != ""), 2) if any(r["attempts"] != "" for r in rows) else None,
            "latency_ms_mean": round(statistics.mean(latencies), 2) if latencies else None,
            "latency_ms_max": max(latencies) if latencies else None,
        },
    }
    (RESULTS / "builder_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "providers"}, indent=2))
    for r in rows:
        if not r["correct"]:
            print("FAILED:", r["scenario"], r["instruction"], "->", r["error"] or r["ops"])
    print(f"\nWrote {RESULTS / 'builder_edits.csv'} and {RESULTS / 'builder_summary.json'}")


if __name__ == "__main__":
    main()
