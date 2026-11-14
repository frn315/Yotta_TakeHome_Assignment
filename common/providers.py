"""One provider interface for generation and embeddings.

OpenAI, NVIDIA NIM and Llama servers (Ollama, vLLM, llama.cpp) all speak the
OpenAI-compatible /chat/completions and /embeddings protocol, and so do most
gateways. So there is ONE HTTP client; a provider is just an `Endpoint`
(base URL, key, model) chosen through environment variables.

Models differ in which optional parameters they accept (response_format,
temperature, max_tokens vs max_completion_tokens). Instead of a per-model
capability table that goes stale, the client adapts: on a 400 that names a
parameter it drops or renames that parameter, retries, and remembers.
"""
from __future__ import annotations

import json
import re
import time
import zlib
from typing import Any, Protocol

import httpx
import numpy as np

from .config import LOCAL_EMBEDDINGS, Endpoint, Settings


class ProviderError(Exception):
    """A provider problem that can be shown to the user.

    kind: missing_credentials | config | auth | model_not_found | rate_limit
          | unavailable | timeout | bad_request | bad_response
    """

    def __init__(self, kind: str, message: str, provider: str = ""):
        super().__init__(message)
        self.kind, self.message, self.provider = kind, message, provider

    @property
    def http_status(self) -> int:
        """Status an API should answer with: 503 = fix the setup or wait, 504 = too slow, 502 = the provider misbehaved."""
        return {"missing_credentials": 503, "config": 503, "rate_limit": 503, "unavailable": 503, "timeout": 504}.get(self.kind, 502)


class ChatClient(Protocol):
    provider: str
    model: str

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str: ...


class EmbeddingClient(Protocol):
    provider: str
    model: str

    def embed(self, texts: list[str], input_type: str = "passage") -> np.ndarray: ...


class _HTTPClient:
    def __init__(self, endpoint: Endpoint, timeout_s: float, max_retries: int,
                 transport: httpx.BaseTransport | None = None):
        if endpoint.key_required and not endpoint.api_key:
            raise ProviderError("missing_credentials",
                                f"{endpoint.key_env} is not set. Add it to your environment or .env (see .env.example).",
                                endpoint.provider)
        if not endpoint.base_url or not endpoint.model:
            raise ProviderError("config", f"Provider '{endpoint.provider}' needs a base URL and a model. Use openai, "
                                "nim or llama, or set *_BASE_URL and *_MODEL for another OpenAI-compatible endpoint.",
                                endpoint.provider)
        self.provider, self.model, self.max_retries = endpoint.provider, endpoint.model, max(0, max_retries)
        headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
        self._http = httpx.Client(base_url=endpoint.base_url, headers=headers, timeout=timeout_s, transport=transport)

    def _post(self, path: str, payload: dict) -> dict:
        error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(2 ** attempt, 8))
            try:
                resp = self._http.post(path, json=payload)
            except httpx.TimeoutException:
                error = ProviderError("timeout", f"{self.provider} timed out. Raise LLM_TIMEOUT_S for slow models.", self.provider)
                continue
            except httpx.HTTPError as exc:
                error = ProviderError("unavailable", f"Could not reach {self.provider} at {self._http.base_url} "
                                      f"({type(exc).__name__}). Check the base URL and that the server is running.", self.provider)
                continue
            detail = resp.text[:300]
            if resp.status_code < 400:
                try:
                    return resp.json()
                except ValueError:
                    raise ProviderError("bad_response", f"{self.provider} returned non-JSON: {detail}", self.provider)
            if resp.status_code in (401, 403):
                raise ProviderError("auth", f"Authentication failed ({resp.status_code}). Check the API key. {detail}", self.provider)
            if resp.status_code == 404:
                raise ProviderError("model_not_found", f"Model or endpoint not found ('{self.model}'). {detail}", self.provider)
            if resp.status_code in (400, 422):
                raise ProviderError("bad_request", detail, self.provider)
            error = ProviderError("rate_limit" if resp.status_code == 429 else "unavailable",
                                  f"{self.provider} returned HTTP {resp.status_code}: {detail}", self.provider)
        raise error  # retries exhausted


class OpenAICompatChat(_HTTPClient):
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        super().__init__(settings.llm, settings.llm_timeout_s, settings.llm_max_retries, transport)
        self._params: dict[str, Any] = {"temperature": settings.llm_temperature, "max_tokens": settings.llm_max_tokens}
        self._json_mode_ok = True

    def _adapt(self, payload: dict, error: str) -> bool:
        """Remove/rename the parameter a 400 complains about. Returns True if the payload changed.

        The client is shared between request threads, so two calls can learn the same thing at
        once: the remembered state is rebuilt from the payload, never popped from, which makes
        learning it twice harmless."""
        error = error.lower()
        if "max_completion_tokens" in error and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        elif "response_format" in payload and ("response_format" in error or "json" in error):
            payload.pop("response_format")
            self._json_mode_ok = False
        elif "temperature" in error and "temperature" in payload:
            payload.pop("temperature")
        else:
            return False
        self._params = {k: v for k, v in payload.items() if k not in ("model", "messages", "response_format")}
        return True

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, **self._params}
        if json_mode and self._json_mode_ok:
            payload["response_format"] = {"type": "json_object"}
        while True:
            try:
                data = self._post("/chat/completions", payload)
                break
            except ProviderError as exc:
                if exc.kind != "bad_request" or not self._adapt(payload, exc.message):
                    raise
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("bad_response", f"Unexpected chat response: {str(data)[:200]}", self.provider)
        if not content.strip():
            raise ProviderError("bad_response", "Model returned an empty message "
                                "(reasoning models may need a larger LLM_MAX_TOKENS).", self.provider)
        return content


class OpenAICompatEmbeddings(_HTTPClient):
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        super().__init__(settings.embedding, settings.llm_timeout_s, settings.llm_max_retries, transport)

    def embed(self, texts: list[str], input_type: str = "passage") -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), 32):
            payload: dict[str, Any] = {"model": self.model, "input": texts[i:i + 32]}
            if self.provider == "nim":   # NIM retrieval embedders are asymmetric (query vs passage)
                payload |= {"input_type": "query" if input_type == "query" else "passage", "truncate": "END"}
            data = self._post("/embeddings", payload)
            try:
                rows += [r["embedding"] for r in sorted(data["data"], key=lambda r: r.get("index", 0))]
            except (KeyError, TypeError):
                raise ProviderError("bad_response", f"Unexpected embeddings response: {str(data)[:200]}", self.provider)
        return _l2_normalise(np.asarray(rows, dtype=np.float32))


def _l2_normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashEmbeddings:
    """Offline feature-hashing embedder (word uni/bigrams + character trigrams).

    For chat gateways that expose no /embeddings endpoint, and for tests. It
    captures lexical and sub-word overlap only, NOT meaning: treat its dense
    numbers as a floor, not as a result about dense retrieval.
    """

    provider, model = LOCAL_EMBEDDINGS, "hash-1024"
    _token = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _features(self, text: str):
        words = self._token.findall(text.lower())
        for w in words:
            yield "w:" + w, 1.0
            padded = f"#{w}#"
            for i in range(len(padded) - 2):
                yield "c:" + padded[i:i + 3], 0.3
        for a, b in zip(words, words[1:]):
            yield f"b:{a}_{b}", 0.7

    def embed(self, texts: list[str], input_type: str = "passage") -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                h = zlib.crc32(feature.encode())
                m[row, h % self.dim] += weight if (h >> 16) & 1 else -weight   # signed hashing
        return _l2_normalise(m)


def get_chat(settings: Settings, transport: httpx.BaseTransport | None = None) -> ChatClient:
    return OpenAICompatChat(settings, transport)


def get_embedder(settings: Settings, transport: httpx.BaseTransport | None = None) -> EmbeddingClient:
    if settings.embedding.provider == LOCAL_EMBEDDINGS:
        return HashEmbeddings()
    return OpenAICompatEmbeddings(settings, transport)


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating reasoning tags, code fences and prose."""
    text = _THINK.sub("", text).strip()
    fenced = _FENCE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except (ValueError, TypeError):
            continue
    raise ValueError("Model reply did not contain valid JSON.")
