"""Structured edit operations (patches).

The edit agent never rewrites the site. It emits a short list of operations;
each knows how to apply itself to a site dict. `apply_ops` runs them on a deep
copy and re-validates, so the input is never mutated, a failing edit leaves no
partial state, and everything an edit does not touch is preserved by construction.
"""
from __future__ import annotations

import copy
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .schema import SECTION_TYPES, SectionType, Site, default_props

MAX_OPS = 8


class OpError(Exception):
    """An operation that cannot be applied to the current site (bad id, bad path...)."""


def _index(site: dict, section_id: str) -> int:
    for i, section in enumerate(site["sections"]):
        if section["id"] == section_id:
            return i
    raise OpError(f"No section with id '{section_id}'. Available ids: {[s['id'] for s in site['sections']]}")


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def apply(self, site: dict) -> dict:  # returns a change record for the log
        raise NotImplementedError


class UpdateContent(_Op):
    op: Literal["update_content"]
    section_id: str
    path: str            # dotted path inside props: "headline", "plans.0.price"
    value: Any

    def apply(self, site: dict) -> dict:
        node: Any = site["sections"][_index(site, self.section_id)]["props"]
        # Models often write "props.headline": the path is already relative to props, and no section has
        # a prop called "props", so the prefix is dropped instead of spending repair attempts on it.
        parts = [p for p in self.path.removeprefix("props.").split(".") if p]
        if not parts:
            raise OpError("Empty path.")
        for depth, part in enumerate(parts):
            # The path must already exist, so a typo fails loudly instead of adding a stray key.
            if isinstance(node, list) and part.isdigit() and int(part) < len(node):
                key: Any = int(part)
            elif isinstance(node, dict) and part in node:
                key = part
            else:
                have = (f"indexes 0..{len(node) - 1}" if isinstance(node, list)
                        else f"fields {sorted(node)}" if isinstance(node, dict) else "no children")
                raise OpError(f"Path '{self.path}' does not exist in section '{self.section_id}': '{part}' not found ({have}).")
            if depth < len(parts) - 1:
                node = node[key]
        old, node[key] = node[key], self.value
        return {"section_id": self.section_id, "path": self.path, "old": old, "new": self.value}


class UpdateStyle(_Op):
    op: Literal["update_style"]
    section_id: str
    property: Literal["background", "text_color", "padding", "align"]
    value: Optional[str]

    def apply(self, site: dict) -> dict:
        style = site["sections"][_index(site, self.section_id)]["style"]
        old, style[self.property] = style.get(self.property), self.value
        return {"section_id": self.section_id, "property": self.property, "old": old, "new": self.value}


class UpdateTheme(_Op):
    op: Literal["update_theme"]
    property: Literal["primary", "background", "text", "font"]
    value: str

    def apply(self, site: dict) -> dict:
        old, site["theme"][self.property] = site["theme"].get(self.property), self.value
        return {"property": self.property, "old": old, "new": self.value}


class UpdateTitle(_Op):
    op: Literal["update_title"]
    value: str

    def apply(self, site: dict) -> dict:
        old, site["title"] = site["title"], self.value
        return {"old": old, "new": self.value}


class AddSection(_Op):
    op: Literal["add_section"]
    section_type: SectionType
    after_id: Optional[str] = None            # insert after this section; else `index`; else append
    index: Optional[int] = Field(default=None, ge=0)
    props: Optional[dict[str, Any]] = None    # merged over starter content, so the result is always renderable

    def apply(self, site: dict) -> dict:
        taken = {s["id"] for s in site["sections"]}
        new_id = next(i for i in [self.section_type, *(f"{self.section_type}-{n}" for n in range(2, 99))] if i not in taken)
        if self.after_id is not None:
            at = _index(site, self.after_id) + 1
        else:
            at = len(site["sections"]) if self.index is None else min(self.index, len(site["sections"]))
        props = {**default_props(self.section_type, site["title"]), **(self.props or {})}
        site["sections"].insert(at, {"id": new_id, "type": self.section_type, "props": props, "style": {}})
        return {"section_id": new_id, "index": at}


class RemoveSection(_Op):
    op: Literal["remove_section"]
    section_id: str

    def apply(self, site: dict) -> dict:
        at = _index(site, self.section_id)
        if len(site["sections"]) == 1:
            raise OpError("Cannot remove the only section on the page.")
        site["sections"].pop(at)
        return {"section_id": self.section_id, "index": at}


class MoveSection(_Op):
    op: Literal["move_section"]
    section_id: str
    to_index: int = Field(ge=0)               # final position, 0 = top

    def apply(self, site: dict) -> dict:
        at = _index(site, self.section_id)
        to = min(self.to_index, len(site["sections"]) - 1)
        site["sections"].insert(to, site["sections"].pop(at))
        return {"section_id": self.section_id, "from": at, "to": to}


Op = Annotated[
    Union[UpdateContent, UpdateStyle, UpdateTheme, UpdateTitle, AddSection, RemoveSection, MoveSection],
    Field(discriminator="op"),
]
OPS_ADAPTER: TypeAdapter[list[Op]] = TypeAdapter(list[Op])


def apply_ops(site: dict, ops: list[_Op]) -> tuple[dict, list[dict]]:
    """Apply ops to a deep copy of `site`; return (validated new site, change records).

    Raises OpError for un-appliable ops and pydantic.ValidationError if the result
    breaks the Site schema. All-or-nothing: the caller's `site` is never modified.
    """
    if not ops:
        raise OpError("No operations to apply.")
    if len(ops) > MAX_OPS:
        raise OpError(f"Too many operations ({len(ops)}); the limit is {MAX_OPS} per edit.")
    draft = copy.deepcopy(site)
    changes = [{"op": op.op, **op.apply(draft)} for op in ops]
    return Site.model_validate(draft).model_dump(), changes


OPS_PROMPT = (
    "Allowed operations (emit the smallest list that satisfies the request):\n"
    '- {"op":"update_content","section_id":ID,"path":"dotted.path.in.props","value":ANY}  paths like "headline", "plans.0.price", "items.1.quote"; to add or delete a list item, set the whole list (path "items" or "plans")\n'
    '- {"op":"update_style","section_id":ID,"property":"background"|"text_color"|"padding"|"align","value":STR}\n'
    '- {"op":"update_theme","property":"primary"|"background"|"text"|"font","value":STR}\n'
    '- {"op":"update_title","value":STR}\n'
    f'- {{"op":"add_section","section_type":one of {list(SECTION_TYPES)},"after_id":ID|null,"index":INT|null,"props":OBJECT|null}}\n'
    '- {"op":"remove_section","section_id":ID}\n'
    '- {"op":"move_section","section_id":ID,"to_index":INT}  to_index is the final position, 0 = top'
)
