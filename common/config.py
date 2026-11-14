"""Configuration: environment variables (or .env) only. No secrets in code.

A provider is nothing more than (base_url, api_key, model). `openai`, `nim` and
`llama` are presets for those three values; any other OpenAI-compatible gateway
works by overriding LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PRESETS = {
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "key_required": True,
               "chat_model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small"},
    "nim": {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY", "key_required": True,
            "chat_model": "nvidia/nemotron-3-nano-30b-a3b", "embedding_model": "nvidia/nv-embedqa-e5-v5"},
    # Any OpenAI-compatible Llama server: Ollama (default), llama.cpp, vLLM, LM Studio.
    "llama": {"base_url": "http://localhost:11434/v1", "key_env": "LLAMA_API_KEY", "key_required": False,
              "chat_model": "llama3.1:8b", "embedding_model": "nomic-embed-text"},
}
LOCAL_EMBEDDINGS = "hash"   # offline lexical embedder, for gateways without an /embeddings endpoint


def load_dotenv(path: Path | None = None) -> None:
    """Minimal KEY=VALUE loader. Real environment variables take precedence."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.split(" #")[0].strip().strip("\"'"))


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _cast(raw: str, default):
    """Convert an environment string to the type of the field's default; bad numbers keep the default."""
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(default, Path):
        return Path(raw)
    if isinstance(default, str):
        return raw.lower()
    try:
        return type(default)(raw)
    except ValueError:
        return default


@dataclass
class Endpoint:
    provider: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    key_env: str = ""
    key_required: bool = False

    @classmethod
    def from_env(cls, prefix: str, provider: str, model_field: str) -> "Endpoint":
        preset = PRESETS.get(provider, {})
        key_env = preset.get("key_env", f"{prefix}_API_KEY")
        return cls(
            provider=provider,
            base_url=_env(f"{prefix}_BASE_URL", preset.get("base_url", "")).rstrip("/"),
            model=_env(f"{prefix}_MODEL", preset.get(model_field, "")),
            api_key=_env(f"{prefix}_API_KEY") or _env(key_env),     # generic override, then provider key
            key_env=key_env,
            key_required=preset.get("key_required", False),
        )

    def public(self) -> dict:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url, "key_set": bool(self.api_key)}


@dataclass
class Settings:
    llm: Endpoint
    embedding: Endpoint
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 2           # transport retries for 429 / 5xx / timeouts
    builder_max_repairs: int = 2       # schema-repair attempts after the first try
    rag_num_queries: int = 50
    rag_chunk_words: int = 40
    rag_chunk_overlap: int = 10
    rag_top_k: int = 5
    rag_retriever: str = "hybrid"      # dense | bm25 | hybrid
    rag_chunker: str = "sentence"      # fixed | sentence
    rag_grader: str = "threshold"      # threshold | llm
    rag_grade_threshold: float = 0.5
    rag_max_rewrites: int = 1
    rag_force_local: bool = False
    log_dir: Path = ROOT / "logs"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        llm = _env("LLM_PROVIDER", "nim").lower()
        emb = _env("EMBEDDING_PROVIDER", llm).lower()
        # Every other field is read from the variable of the same name in upper case
        # (RAG_TOP_K, LOG_DIR, ...); the dataclass defaults above are the only defaults.
        tuning = {f.name: _cast(raw, f.default) for f in fields(cls)
                  if f.name not in ("llm", "embedding") and (raw := _env(f.name.upper()))}
        return cls(llm=Endpoint.from_env("LLM", llm, "chat_model"),
                   embedding=Endpoint.from_env("EMBEDDING", emb, "embedding_model"), **tuning)

    def public(self) -> dict:
        """Safe-to-display view (never includes keys)."""
        return {"llm": self.llm.public(), "embedding": self.embedding.public()}
