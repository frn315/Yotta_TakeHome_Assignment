"""Agent workflow: Generator and Edit agent, each wrapped in a validate-and-repair loop.

Roles
-----
Generator   brief -> {"plan": [...], "site": {...}}. Planning and writing are ONE
            model call: the plan is emitted first (so the model commits to a
            structure before writing copy, and the decision is logged), then the
            site. A separate planner call would double latency and add a second
            failure point for a one-page site.
Edit agent  (current site, request) -> patch operations (builder/ops.py).
Validator   plain code, not an agent: Pydantic schema + apply_ops. Deterministic
            checks should not be delegated to a model.

Stopping / error conditions
---------------------------
* Each step gets 1 + max_repairs attempts; validation errors are fed back verbatim.
  After that: WorkflowError, state unchanged.
* The edit agent may decline ("error" set, no ops): terminal, no retry.
* Empty / over-long input is rejected before any model call.
* Provider failures propagate as ProviderError (not retried here; the HTTP client
  already retries transient ones).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from pydantic import ValidationError

from common.events import EventLog
from common.providers import ChatClient, extract_json

from .ops import OPS_ADAPTER, OPS_PROMPT, OpError, apply_ops
from .schema import SCHEMA_PROMPT, SECTION_TYPES, Site

MAX_INPUT_CHARS = 2000
LOG_CHARS = 4000        # prompt / reply text kept per llm_call event

GENERATE_SYSTEM = (
    "You build one-page websites for a no-code website builder. First plan the page, then write it.\n"
    f"Allowed section types: {list(SECTION_TYPES)}. Start with a hero, include every section the brief asks for, "
    "3-8 sections in a sensible order. Write specific, realistic copy for this business; no lorem ipsum.\n"
    + SCHEMA_PROMPT
    + '\nReply with JSON only: {"plan": [{"type": str, "purpose": str}], '
      '"site": {"title": str, "theme": {...}, "sections": [Section, ...]}}'
)
EDIT_SYSTEM = (
    "You are the edit agent of a no-code website builder. You receive the current website JSON and one user "
    "request. Do NOT rewrite the website. Find the target section and property and emit the minimal patch.\n"
    + OPS_PROMPT + "\n" + SCHEMA_PROMPT + "\n"
    "Rules: use existing section ids exactly. For relative requests ('darker', 'bigger') compute a concrete new "
    "value from the current one, and keep text readable (also set text_color when a background change would hurt "
    "contrast). If the request is unclear, impossible within the schema, or targets something that does not "
    "exist, return no ops and say why in 'error'.\n"
    'Reply with JSON only: {"ops": [...], "explanation": "one short sentence for the user", "error": str|null}'
)


class WorkflowError(Exception):
    """The workflow stopped without a valid result. `reason` is safe to show to the user."""

    def __init__(self, reason: str, code: str):
        super().__init__(reason)
        self.reason, self.code = reason, code


@dataclass
class EditResult:
    site: dict
    ops: list[dict]
    changes: list[dict]
    explanation: str
    attempts: int
    latency_ms: float


def _describe(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:8])
    return str(exc)


def normalise_ids(site: dict) -> dict:
    """Deterministic pre-validation repair: unique kebab-case ids. Cheaper than a model round-trip
    for the most common generation defect."""
    seen: set[str] = set()
    sections = site.get("sections") if isinstance(site, dict) else None
    for section in sections if isinstance(sections, list) else []:
        if not isinstance(section, dict):
            continue
        base = re.sub(r"[^a-z0-9]+", "-", str(section.get("id") or section.get("type") or "section").lower()).strip("-")
        base = (base if base[:1].isalpha() else f"s-{base}")[:36]
        sid = next(c for c in [base, *(f"{base}-{n}" for n in range(2, 99))] if c not in seen)
        seen.add(sid)
        section["id"] = sid
    return site


class Workflow:
    def __init__(self, chat: ChatClient, log: EventLog, max_repairs: int = 2):
        self.chat, self.log, self.max_attempts = chat, log, 1 + max(0, max_repairs)

    def _repair_loop(self, step: str, system: str, user: str, accept):
        """Call the model, hand its JSON to `accept`, and on a validation problem retry with the
        error as feedback. `accept` returns the result or raises ValueError/OpError."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        problem = ""
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            reply = self.chat.complete(messages, json_mode=True)
            self.log.log("llm_call", step=step, attempt=attempt, provider=self.chat.provider, model=self.chat.model,
                         latency_ms=round((time.perf_counter() - started) * 1000, 1),
                         sent=messages[-1]["content"][:LOG_CHARS], reply=reply[:LOG_CHARS])
            try:
                data = extract_json(reply)
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")
                return accept(data), attempt
            except (ValueError, OpError) as exc:      # pydantic.ValidationError is a ValueError
                problem = _describe(exc)
                self.log.log("validation_failed", step=step, attempt=attempt, errors=problem)
                messages = messages[:2] + [
                    {"role": "assistant", "content": reply[:4000]},
                    {"role": "user", "content": f"That reply was rejected: {problem}\nReturn corrected JSON only."},
                ]
        self.log.log(f"{step}_failed", attempts=self.max_attempts, errors=problem)
        raise WorkflowError(f"No valid result after {self.max_attempts} attempts. Last problem: {problem}", f"{step}_invalid")

    def _check_input(self, text: str, what: str) -> str:
        text = text.strip()
        if not text:
            raise WorkflowError(f"Please enter {what}.", "empty_input")
        if len(text) > MAX_INPUT_CHARS:
            raise WorkflowError(f"That is too long; keep it under {MAX_INPUT_CHARS} characters.", "input_too_long")
        return text

    def generate(self, brief: str) -> dict:
        brief = self._check_input(brief, "a description of the website")
        self.log.log("generate_started", brief=brief[:200])

        def accept(data: dict) -> dict:
            site = Site.model_validate(normalise_ids(data.get("site", data))).model_dump()
            plan = data.get("plan") if isinstance(data.get("plan"), list) else []
            self.log.log("plan", sections=[p.get("type") for p in plan if isinstance(p, dict)])
            return site

        site, attempt = self._repair_loop("generate", GENERATE_SYSTEM, f"Brief: {brief}", accept)
        self.log.log("site_generated", attempt=attempt, sections=[s["id"] for s in site["sections"]])
        return site

    def edit(self, site: dict, instruction: str) -> EditResult:
        instruction = self._check_input(instruction, "the change you want")
        started = time.perf_counter()
        self.log.log("edit_started", instruction=instruction)

        def accept(data: dict):
            raw_ops = data.get("ops") or []
            if data.get("error") and not raw_ops:
                self.log.log("edit_declined", reason=str(data["error"]))
                raise WorkflowError(str(data["error"]), "edit_declined")
            ops = OPS_ADAPTER.validate_python(raw_ops)
            new_site, changes = apply_ops(site, ops)
            if new_site == site:
                raise OpError("These operations change nothing. Emit ops that implement the request, or set 'error'.")
            return new_site, ops, changes, str(data.get("explanation") or "Done.")

        user = f"Current website JSON:\n{json.dumps(site, separators=(',', ':'))}\n\nUser request: {instruction}"
        (new_site, ops, changes, explanation), attempt = self._repair_loop("edit", EDIT_SYSTEM, user, accept)
        for change in changes:
            self.log.log("edit_operation", **change)
        self.log.log("edit_applied", attempt=attempt, ops=len(ops))
        return EditResult(new_site, [o.model_dump() for o in ops], changes, explanation, attempt,
                          round((time.perf_counter() - started) * 1000, 1))
