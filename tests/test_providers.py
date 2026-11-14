"""Provider layer: one client, any OpenAI-compatible endpoint, graceful failures."""
import json

import httpx
import pytest

from common import providers
from common.config import Settings
from common.providers import ProviderError, extract_json, get_chat, get_embedder

USER = [{"role": "user", "content": "hi"}]
OK = {"choices": [{"message": {"content": "ok"}}]}


def from_env(monkeypatch, **env) -> Settings:
    for key in ("OPENAI_API_KEY", "NVIDIA_API_KEY", "LLAMA_API_KEY", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL",
                "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "LLM_MAX_RETRIES"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("common.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    return Settings.from_env()


def chat_with(settings, handler):
    return get_chat(settings, transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("env,host,model", [
    ({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "k"}, "api.openai.com", "gpt-4o-mini"),
    ({"LLM_PROVIDER": "nim", "NVIDIA_API_KEY": "k"}, "integrate.api.nvidia.com", "nvidia/nemotron-3-nano-30b-a3b"),
    ({"LLM_PROVIDER": "llama", "LLAMA_API_KEY": "k"}, "localhost:11434", "llama3.1:8b"),
    # any other OpenAI-compatible gateway: preset + overrides, or a free-form provider name
    ({"LLM_PROVIDER": "openai", "LLM_BASE_URL": "https://gateway.test/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m-1"}, "gateway.test", "m-1"),
    ({"LLM_PROVIDER": "acme", "LLM_BASE_URL": "https://acme.test/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m-2"}, "acme.test", "m-2"),
])
def test_every_provider_goes_through_the_same_client(monkeypatch, env, host, model):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(url=str(request.url), auth=request.headers.get("authorization"), body=json.loads(request.content))
        return httpx.Response(200, json=OK)

    assert chat_with(from_env(monkeypatch, **env), handler).complete(USER, json_mode=True) == "ok"
    assert host in seen["url"] and seen["url"].endswith("/v1/chat/completions")
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == model and seen["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("env,kind,hint", [
    ({"LLM_PROVIDER": "openai"}, "missing_credentials", "OPENAI_API_KEY"),
    ({"LLM_PROVIDER": "nim"}, "missing_credentials", "NVIDIA_API_KEY"),
    ({"LLM_PROVIDER": "banana"}, "config", "base URL"),
])
def test_misconfiguration_is_a_clear_error(monkeypatch, env, kind, hint):
    with pytest.raises(ProviderError) as err:
        get_chat(from_env(monkeypatch, **env))
    assert err.value.kind == kind and hint in err.value.message


def test_llama_needs_no_key(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=OK)

    assert chat_with(from_env(monkeypatch, LLM_PROVIDER="llama"), handler).complete(USER) == "ok"
    assert seen["auth"] is None


def test_http_failures_map_to_error_kinds(monkeypatch):
    settings = from_env(monkeypatch, LLM_PROVIDER="llama", LLM_MAX_RETRIES="1")

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    def slow(request):
        raise httpx.ReadTimeout("too slow")

    cases = [(refuse, "unavailable"), (slow, "timeout"),
             (lambda r: httpx.Response(200, text="<html>login</html>"), "bad_response"),
             (lambda r: httpx.Response(200, json={"choices": [{"message": {"content": None}}]}), "bad_response"),
             (lambda r: httpx.Response(200, json={"error": "x"}), "bad_response")]
    cases += [(lambda r, s=s: httpx.Response(s, text="nope"), k)
              for s, k in [(401, "auth"), (403, "auth"), (404, "model_not_found"), (400, "bad_request"),
                           (429, "rate_limit"), (503, "unavailable")]]
    for handler, kind in cases:
        with pytest.raises(ProviderError) as err:
            chat_with(settings, handler).complete(USER)
        assert err.value.kind == kind


def test_transient_errors_are_retried_and_permanent_ones_are_not(monkeypatch):
    settings = from_env(monkeypatch, LLM_PROVIDER="llama", LLM_MAX_RETRIES="2")
    calls = []

    def flaky(request):
        calls.append(1)
        return httpx.Response(500, text="boom") if len(calls) == 1 else httpx.Response(200, json=OK)

    assert chat_with(settings, flaky).complete(USER) == "ok" and len(calls) == 2

    calls.clear()

    def unauthorised(request):
        calls.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(ProviderError):
        chat_with(settings, unauthorised).complete(USER)
    assert len(calls) == 1


def test_client_adapts_to_parameters_a_model_rejects_and_remembers(monkeypatch):
    """Different models reject different optional parameters. The client drops/renames whatever a
    400 names, retries, and does not send it again."""
    bodies = []

    def picky(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "max_tokens" in body:
            return httpx.Response(400, text="Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.")
        if "temperature" in body:
            return httpx.Response(400, text="Unsupported value: 'temperature' does not support 0.2 with this model.")
        if "response_format" in body:
            return httpx.Response(400, text="response_format json_object is not supported by this model")
        return httpx.Response(200, json=OK)

    chat = chat_with(from_env(monkeypatch, LLM_PROVIDER="llama"), picky)
    assert chat.complete(USER, json_mode=True) == "ok"
    assert len(bodies) == 4 and set(bodies[-1]) == {"model", "messages", "max_completion_tokens"}
    assert chat.complete(USER, json_mode=True) == "ok" and len(bodies) == 5      # learned: no more failed attempts

    def always_400(request):
        return httpx.Response(400, text="context length exceeded")

    with pytest.raises(ProviderError) as err:                                        # unrelated 400: no retry loop
        chat_with(from_env(monkeypatch, LLM_PROVIDER="llama"), always_400).complete(USER)
    assert err.value.kind == "bad_request"


def test_embeddings_batching_order_and_nim_input_type(monkeypatch):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        rows = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": rows[::-1]})                         # out of order on purpose

    settings = from_env(monkeypatch, LLM_PROVIDER="nim", NVIDIA_API_KEY="k")
    emb = get_embedder(settings, transport=httpx.MockTransport(handler))
    vectors = emb.embed(["a" * (i + 1) for i in range(40)], input_type="passage")
    assert vectors.shape == (40, 2) and [len(b["input"]) for b in bodies] == [32, 8]
    assert vectors[0, 0] < vectors[39, 0]                                             # order restored by index
    emb.embed(["q"], input_type="query")
    assert [b["input_type"] for b in bodies] == ["passage", "passage", "query"]

    plain = from_env(monkeypatch, LLM_PROVIDER="openai", OPENAI_API_KEY="k")
    get_embedder(plain, transport=httpx.MockTransport(handler)).embed(["x"])
    assert "input_type" not in bodies[-1]                                             # NIM-only parameter


def test_chat_and_embeddings_can_use_different_providers(monkeypatch):
    settings = from_env(monkeypatch, LLM_PROVIDER="openai", LLM_BASE_URL="https://gateway.test/v1", LLM_API_KEY="k",
                        EMBEDDING_PROVIDER="hash")
    emb = get_embedder(settings)
    a, b = emb.embed(["solar system planets"]), emb.embed(["solar system planets"])
    assert emb.provider == "hash" and (a == b).all() and abs(float((a ** 2).sum()) - 1.0) < 1e-5


def test_keys_never_appear_in_public_settings_or_repr(monkeypatch):
    settings = from_env(monkeypatch, LLM_PROVIDER="openai", OPENAI_API_KEY="sk-secret-123")
    assert "sk-secret-123" not in json.dumps(settings.public()) + repr(settings)
    assert settings.public()["llm"]["key_set"] is True


@pytest.mark.parametrize("text", [
    '{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here it is: {"a": 1} Hope that helps.', '<think>hmm {x}</think>{"a": 1}',
])
def test_extract_json_tolerates_common_model_noise(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_two_threads_learning_the_same_rejected_parameter_do_not_crash(monkeypatch):
    """The chat client is shared by request threads; both may get the same 400 before either adapts."""
    chat = chat_with(from_env(monkeypatch, LLM_PROVIDER="llama"), lambda r: httpx.Response(200, json=OK))
    first, second = ({"model": "m", "messages": [], "temperature": 0.2, "max_tokens": 10} for _ in range(2))
    for error in ("'temperature' is not supported", "use 'max_completion_tokens' instead"):
        assert chat._adapt(first, error) and chat._adapt(second, error)
    assert first == second == {"model": "m", "messages": [], "max_completion_tokens": 10}
    assert chat._params == {"max_completion_tokens": 10}


def test_tuning_variables_are_cast_to_the_type_of_their_default(monkeypatch):
    settings = from_env(monkeypatch, LLM_PROVIDER="llama", RAG_TOP_K="3", RAG_FORCE_LOCAL="true", RAG_RETRIEVER="BM25",
                        LLM_TEMPERATURE="not-a-number", LLM_MAX_RETRIES="-5")
    assert (settings.rag_top_k, settings.rag_force_local, settings.rag_retriever) == (3, True, "bm25")
    assert settings.llm_temperature == 0.2                                            # bad number keeps the default
    assert chat_with(settings, lambda r: httpx.Response(200, json=OK)).complete(USER) == "ok"   # negative retries
