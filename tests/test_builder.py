"""Workstream B: schema contract, patch operations, agent workflow, API."""
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from builder.agents import Workflow, WorkflowError
from builder.app import create_app
from builder.ops import OPS_ADAPTER, OpError, apply_ops
from builder.schema import Site
from common.events import EventLog
from common.providers import get_chat
from tests.conftest import ScriptedChat


def ops(*raw):
    return OPS_ADAPTER.validate_python(list(raw))


def reply(*raw_ops, explanation="Done.", error=None):
    return json.dumps({"ops": list(raw_ops), "explanation": explanation, "error": error})


def section(site, sid):
    return next(s for s in site["sections"] if s["id"] == sid)


DARKER = {"op": "update_style", "section_id": "hero", "property": "background", "value": "#0b1a45"}
CTA = {"op": "update_content", "section_id": "hero", "path": "cta_label", "value": "Book a Trial"}
ADD = {"op": "add_section", "section_type": "features", "after_id": "hero"}


# ---------------------------------------------------------------- schema
def test_schema_rejects_what_the_renderer_cannot_draw(site):
    for mutate in (
        lambda s: s["sections"][0]["style"].update(background="dark blue"),      # colour name
        lambda s: s["sections"][0]["props"].update(sparkles=True),               # unknown prop
        lambda s: s["sections"][0].update(type="carousel"),                      # unknown component
        lambda s: s["sections"][1].update(id="hero"),                            # duplicate id
        lambda s: s["sections"][1]["props"].update(plans=[]),                    # empty pricing table
        lambda s: s.update(sections=[]),                                         # empty page
    ):
        bad = copy.deepcopy(site)
        mutate(bad)
        with pytest.raises(ValidationError):
            Site.model_validate(bad)
    assert Site.model_validate({"title": "T", "sections": [{"id": "a", "type": "hero", "props": {"headline": "H"}}],
                                "theme": {"primary": "#ABC"}}).theme.primary == "#aabbcc"   # normalised


# ---------------------------------------------------------------- patch ops
def test_ops_are_all_or_nothing_and_touch_only_the_target(site):
    original = copy.deepcopy(site)
    new, changes = apply_ops(site, ops(CTA))
    assert site == original                                                      # input never mutated
    assert section(new, "hero")["props"]["cta_label"] == "Book a Trial"
    assert [s for s in new["sections"] if s["id"] != "hero"] == site["sections"][1:]
    assert changes == [{"op": "update_content", "section_id": "hero", "path": "cta_label", "old": "Get started", "new": "Book a Trial"}]

    with pytest.raises(OpError):                                                 # 2nd op fails -> 1st must not leak
        apply_ops(site, ops(CTA, {"op": "remove_section", "section_id": "nope"}))
    assert site == original


def test_nested_paths_add_remove_move_and_unique_ids(site):
    new, _ = apply_ops(site, ops({"op": "update_content", "section_id": "pricing", "path": "plans.1.price", "value": "$59"}))
    prefixed, _ = apply_ops(site, ops({"op": "update_content", "section_id": "hero", "path": "props.cta_label", "value": "Go"}))
    assert section(prefixed, "hero")["props"]["cta_label"] == "Go"                # "props." prefix is tolerated
    new, _ = apply_ops(new, ops(ADD, {"op": "add_section", "section_type": "features"}))
    assert [s["id"] for s in new["sections"]] == ["hero", "features", "pricing", "testimonials", "contact", "features-2"]
    new, _ = apply_ops(new, ops({"op": "move_section", "section_id": "contact", "to_index": 0},
                                {"op": "remove_section", "section_id": "features-2"}))
    assert [s["id"] for s in new["sections"]] == ["contact", "hero", "features", "pricing", "testimonials"]
    assert section(new, "pricing")["props"]["plans"][1]["price"] == "$59"         # earlier edit survived
    new, _ = apply_ops(new, ops({"op": "add_section", "section_type": "text", "index": 0, "props": {"heading": "Our story"}}))
    assert new["sections"][0]["props"] == {"heading": "Our story", "body": section(new, "text")["props"]["body"]}


def test_unappliable_ops_fail_with_messages_a_model_can_act_on(site):
    for raw, expected in [
        ({"op": "remove_section", "section_id": "footer"}, "Available ids: ['hero', 'pricing', 'testimonials', 'contact']"),
        ({"op": "update_content", "section_id": "hero", "path": "title", "value": "x"}, "'title' not found (fields ['cta_href'"),
        ({"op": "update_content", "section_id": "pricing", "path": "plans.9.price", "value": "x"}, "'9' not found (indexes 0..1)"),
        ({"op": "update_content", "section_id": "hero", "path": "headline.0", "value": "x"}, "no children"),
        ({"op": "update_content", "section_id": "hero", "path": "", "value": "x"}, "Empty path"),
    ]:
        with pytest.raises(OpError) as err:
            apply_ops(site, ops(raw))
        assert expected in str(err.value)
    with pytest.raises(OpError, match="limit is 8"):
        apply_ops(site, ops(*[CTA] * 9))
    with pytest.raises(ValidationError):                                         # result must satisfy the schema
        apply_ops(site, ops({"op": "update_style", "section_id": "hero", "property": "background", "value": "blue"}))
    with pytest.raises(ValidationError):
        apply_ops(site, ops({"op": "update_content", "section_id": "hero", "path": "headline", "value": ""}))
    with pytest.raises(ValidationError):                                         # unknown op / unknown field
        ops({"op": "delete_everything"})
    only = {**site, "sections": site["sections"][:1]}
    with pytest.raises(OpError, match="only section"):
        apply_ops(only, ops({"op": "remove_section", "section_id": "hero"}))


# ---------------------------------------------------------------- workflow
def test_generation_validates_normalises_ids_and_logs_the_plan():
    good = {"plan": [{"type": "hero", "purpose": "intro"}, {"type": "pricing", "purpose": "plans"}],
            "site": {"title": "Cafe", "sections": [{"id": "Hero 1", "type": "hero", "props": {"headline": "Hi"}},
                                                   {"type": "hero", "props": {"headline": "Again"}}]}}
    bad = {"site": {"title": "Cafe", "sections": [{"id": "h", "type": "hero", "props": {"headline": "Hi", "bogus": 1}}]}}
    log = EventLog("test")
    chat = ScriptedChat(json.dumps(bad), "```json\n" + json.dumps(good) + "\n```")
    site = Workflow(chat, log).generate("a cafe site")
    assert [s["id"] for s in site["sections"]] == ["hero-1", "hero"]
    kinds = [e["kind"] for e in log.events]
    assert kinds == ["generate_started", "llm_call", "validation_failed", "llm_call", "plan", "site_generated"]
    assert "bogus" in chat.calls[1][-1]["content"]                               # the error was fed back
    assert chat.calls[1][-2]["role"] == "assistant"                              # together with the rejected reply


def test_three_edit_types_in_a_row_preserve_earlier_edits(site):
    wf = Workflow(ScriptedChat(reply(DARKER), reply(CTA), reply(ADD)), EventLog("test"))
    v2 = wf.edit(site, "Make the hero section darker").site                      # style
    v3 = wf.edit(v2, "Change the CTA to Book a Trial").site                      # content
    v4 = wf.edit(v3, "Add a features section after the hero").site               # structure
    hero = section(v4, "hero")
    assert hero["style"]["background"] == "#0b1a45" and hero["props"]["cta_label"] == "Book a Trial"
    assert [s["type"] for s in v4["sections"]] == ["hero", "features", "pricing", "testimonials", "contact"]
    assert section(v4, "pricing") == section(site, "pricing")                    # untouched section identical


def test_editor_gets_current_state_and_every_operation_is_logged(site):
    chat, log = ScriptedChat(reply({"op": "update_title", "value": "Iron Temple"}, explanation="Renamed.")), EventLog("test")
    result = Workflow(chat, log).edit(site, "Rename the site to Iron Temple")
    assert result.site["title"] == "Iron Temple" and result.explanation == "Renamed." and result.attempts == 1
    assert json.dumps(site, separators=(",", ":")) in chat.calls[0][1]["content"]
    assert [e["kind"] for e in log.events] == ["edit_started", "llm_call", "edit_operation", "edit_applied"]
    assert log.events[1]["sent"].endswith("Rename the site to Iron Temple") and "update_title" in log.events[1]["reply"]
    assert log.events[2] | {"ts": 0} == {"ts": 0, "source": "test", "kind": "edit_operation", "op": "update_title",
                                         "old": "Fitness Studio", "new": "Iron Temple"}


def test_invalid_model_output_is_repaired_with_feedback(site):
    chat = ScriptedChat(
        "Sure, I'll make it darker!",                                                              # not JSON
        reply({"op": "update_style", "section_id": "header", "property": "background", "value": "#000000"}),  # wrong id
        reply({"op": "update_style", "section_id": "hero", "property": "background", "value": "black"}),      # not hex
        reply({"op": "update_style", "section_id": "hero", "property": "background", "value": "#000000"}),
    )
    log = EventLog("test")
    result = Workflow(chat, log, max_repairs=3).edit(site, "make the hero black")
    assert result.attempts == 4 and section(result.site, "hero")["style"]["background"] == "#000000"
    assert sum(e["kind"] == "validation_failed" for e in log.events) == 3
    assert "Available ids" in chat.calls[2][-1]["content"] and "hex colour" in chat.calls[3][-1]["content"]


def test_stopping_conditions(site):
    garbage = ScriptedChat("nope", "nope", "nope")
    with pytest.raises(WorkflowError) as err:
        Workflow(garbage, EventLog("test"), max_repairs=2).edit(site, "do something")
    assert err.value.code == "edit_invalid" and len(garbage.calls) == 3          # attempt cap

    decline = ScriptedChat(reply(error="There is no footer section."))
    with pytest.raises(WorkflowError) as err:
        Workflow(decline, EventLog("test")).edit(site, "make the footer red")
    assert (err.value.code, err.value.reason, len(decline.calls)) == ("edit_declined", "There is no footer section.", 1)

    noop = ScriptedChat(*[reply({"op": "update_title", "value": site["title"]})] * 3)
    with pytest.raises(WorkflowError, match="change nothing"):                   # a no-op is not a success
        Workflow(noop, EventLog("test")).edit(site, "rename it")

    untouched = ScriptedChat()
    for text, code in [("   ", "empty_input"), ("x" * 2001, "input_too_long")]:
        with pytest.raises(WorkflowError) as err:
            Workflow(untouched, EventLog("test")).edit(site, text)
        assert err.value.code == code
    assert untouched.calls == []                                                 # rejected before any model call


# ---------------------------------------------------------------- API
def test_api_generate_two_edits_bad_edit_stale_version_undo(settings, site):
    chat = ScriptedChat(json.dumps({"plan": [], "site": site}), reply(DARKER), reply(CTA),
                        reply(error="I can't do that."), reply(ADD))
    client = TestClient(create_app(settings, chat))
    created = client.post("/api/sites", json={"brief": "fitness studio"}).json()
    sid = created["site_id"]
    assert created["version"] == 1 and created["site"] == site and created["events"]

    r1 = client.post(f"/api/sites/{sid}/edit", json={"instruction": "darker hero", "base_version": 1}).json()
    r2 = client.post(f"/api/sites/{sid}/edit", json={"instruction": "new cta", "base_version": 2}).json()
    assert (r1["version"], r2["version"]) == (2, 3) and r2["ops"][0]["value"] == "Book a Trial"
    hero = section(r2["site"], "hero")
    assert hero["props"]["cta_label"] == "Book a Trial" and hero["style"]["background"] == "#0b1a45"

    bad = client.post(f"/api/sites/{sid}/edit", json={"instruction": "do a barrel roll"})
    assert bad.status_code == 422 and bad.json()["error"] == {"code": "edit_declined", "message": "I can't do that.", "version": 3}
    assert bad.json()["events"][-1] | {"ts": 0} == {"ts": 0, "source": "builder", "kind": "request_failed", "site_id": sid,
                                                    "code": "edit_declined", "message": "I can't do that."}
    assert client.get(f"/api/sites/{sid}").json() == {"site_id": sid, "version": 3, "site": r2["site"]}   # unchanged

    stale = client.post(f"/api/sites/{sid}/edit", json={"instruction": "add features", "base_version": 1})
    assert stale.status_code == 409 and client.get(f"/api/sites/{sid}").json()["version"] == 3

    undone = client.post(f"/api/sites/{sid}/undo").json()
    assert undone["site"] == r1["site"] and undone["events"][0]["kind"] == "undo"   # v4 = state of v2
    assert client.post(f"/api/sites/{sid}/undo").json()["site"] == site           # v5 = state of v1
    assert client.post(f"/api/sites/{sid}/undo").status_code == 422               # nothing left
    history = client.get(f"/api/sites/{sid}/history").json()["history"]
    assert [h["source"] for h in history] == ["generate", "edit", "edit", "undo", "undo"]
    for path in (f"/api/sites/nope/edit", f"/api/sites/nope/undo"):
        assert client.post(path, json={"instruction": "x"}).status_code == 404
    assert client.get("/").status_code == 200 and "PromptSite" in client.get("/").text


def test_api_through_the_real_http_client_against_a_fake_openai_server(settings, site):
    """Full stack minus the network: FastAPI -> Workflow -> OpenAICompatChat -> (mock transport)."""
    replies = [json.dumps({"plan": [], "site": site}), reply(CTA)]

    def server(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "test-model"
        return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

    client = TestClient(create_app(settings, get_chat(settings, transport=httpx.MockTransport(server))))
    sid = client.post("/api/sites", json={"brief": "fitness studio"}).json()["site_id"]
    edited = client.post(f"/api/sites/{sid}/edit", json={"instruction": "Change the CTA to Book a Trial"}).json()
    assert section(edited["site"], "hero")["props"]["cta_label"] == "Book a Trial"


@pytest.mark.parametrize("status,body,code,http", [
    (401, "bad key", "provider_auth", 502), (429, "slow down", "provider_rate_limit", 503),
    (500, "boom", "provider_unavailable", 503), (404, "no such model", "provider_model_not_found", 502),
])
def test_api_maps_provider_failures_and_keeps_state(settings, site, status, body, code, http):
    ok = ScriptedChat(json.dumps({"site": site}))
    failing = get_chat(settings, transport=httpx.MockTransport(lambda r: httpx.Response(status, text=body)))
    app = create_app(settings, ok)
    client = TestClient(app)
    sid = client.post("/api/sites", json={"brief": "x"}).json()["site_id"]
    app.state.clients["chat"] = failing
    resp = client.post(f"/api/sites/{sid}/edit", json={"instruction": "darker"})
    assert resp.status_code == http and resp.json()["error"]["code"] == code
    assert client.get(f"/api/sites/{sid}").json()["version"] == 1


def test_api_reports_missing_credentials_instead_of_crashing(settings):
    settings.llm.provider, settings.llm.key_required, settings.llm.key_env = "nim", True, "NVIDIA_API_KEY"
    client = TestClient(create_app(settings))
    health = client.get("/api/health").json()
    assert health["ready"] is False and "NVIDIA_API_KEY" in health["problem"]
    resp = client.post("/api/sites", json={"brief": "a cafe"})
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "provider_missing_credentials"


def test_stale_preview_is_refused_before_any_model_call(settings, site):
    chat = ScriptedChat(json.dumps({"site": site}), reply(DARKER))
    client = TestClient(create_app(settings, chat))
    sid = client.post("/api/sites", json={"brief": "x"}).json()["site_id"]
    client.post(f"/api/sites/{sid}/edit", json={"instruction": "darker"})
    stale = client.post(f"/api/sites/{sid}/edit", json={"instruction": "new cta", "base_version": 1})
    assert stale.status_code == 409 and len(chat.calls) == 2                     # generate + first edit only


def test_edit_that_lands_during_a_model_call_is_not_overwritten(settings, site):
    """Two edits computed from v1: the slower one must get 409, not silently replace the faster one."""
    app = create_app(settings, ScriptedChat(json.dumps({"site": site})))
    client = TestClient(app)
    sid = client.post("/api/sites", json={"brief": "x"}).json()["site_id"]

    class SlowChat(ScriptedChat):
        def complete(self, messages, *, json_mode=False):
            other, _ = apply_ops(site, ops(CTA))                                 # another request commits meanwhile
            app.state.store.commit(sid, other, instruction="cta", ops=[CTA], base_version=1)
            return reply(DARKER)

    app.state.clients["chat"] = SlowChat()
    resp = client.post(f"/api/sites/{sid}/edit", json={"instruction": "darker"})      # no base_version sent
    assert resp.status_code == 409
    current = client.get(f"/api/sites/{sid}").json()
    assert current["version"] == 2 and section(current["site"], "hero")["props"]["cta_label"] == "Book a Trial"
