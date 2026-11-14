# Yotta Take-Home Assignment: RAG Pipeline + Agentic Website Builder

Two production-grade prototypes sharing a unified, OpenAI-compatible model layer:

* **Workstream A — RAG Pipeline (`rag/`)**: End-to-end question answering on MS MARCO with dense/BM25/hybrid retrieval, relevance grading, automated query rewriting, and citation-grounded generation.
* **Workstream B — PromptSite (`builder/`)**: Conversational agentic website builder that translates natural-language briefs and edits into deterministic patch operations with live preview, optimistic concurrency, and undo.

Works out of the box with **Ollama (local Llama 3.1)**, **OpenAI**, **NVIDIA NIM**, or any OpenAI-compatible API gateway.

---

## Quick Start

### Option 1: Automatic Setup (macOS / Linux)
Requires [Ollama](https://ollama.com/download) and Python 3.10+.
```bash
./run.sh
```
*Automatically sets up `.venv`, installs dependencies, pulls `llama3.1:8b` and `nomic-embed-text`, and serves both apps at `http://localhost:8000`.*

### Option 2: Manual Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Verify endpoint health and run tests
python -m common.check
pytest -q

# Start server
uvicorn server:create_app --factory --port 8000
```

---

## Applications & Endpoints

Both apps run under a single FastAPI server at `http://localhost:8000`:

| App | Route | Description |
|---|---|---|
| **Website Builder** | `/` | Chat interface to generate and conversationally edit sites with live preview and activity timeline. |
| **Document Chat (RAG)** | `/rag` | Upload custom documents or query the MS MARCO corpus with grounded answers and citation inspection. |

---

## Core Architecture

```text
common/    Unified HTTP provider client (adaptive parameters, backoff retries), config & event logging
rag/       Data cleaning, chunking (fixed/sentence), retrieval (BM25/dense/hybrid), grading, query rewrite, API
builder/   Pydantic schema contracts, patch operations (Command pattern), versioned site store, API
server.py  Single entry point mounting both applications
```

### Key Design Principles
* **Code Owns State, Model Proposes Changes**: The model never mutates HTML or raw state directly. In Builder, edits are validated Pydantic patch operations applied to versioned immutable copies with optimistic concurrency (HTTP 409).
* **Adaptive Provider Layer**: Handles model differences at runtime (automatically stripping or adapting rejected parameters such as `response_format` or `max_tokens`).
* **Resilient RAG**: Combines BM25 and dense embeddings with Reciprocal Rank Fusion (RRF). Irrelevant chunks trigger dynamic query rewriting or explicit abstention rather than hallucinating.

---

## Provider Configuration

Configure via `.env` or environment variables:

| Provider | Setting | Defaults |
|---|---|---|
| **Local Llama** | `LLM_PROVIDER=llama` | `http://localhost:11434/v1` (Ollama), `llama3.1:8b`, `nomic-embed-text` |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`OPENAI_API_KEY=sk-...` | `gpt-4o-mini`, `text-embedding-3-small` |
| **NVIDIA NIM** | `LLM_PROVIDER=nim`<br>`NVIDIA_API_KEY=nvapi-...` | `nvidia/nemotron-3-nano-30b-a3b`, `nvidia/nv-embedqa-e5-v5` |
| **Custom Gateway** | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Any OpenAI-compatible endpoint (vLLM, LM Studio, etc.) |

---

## Evaluation Results

Benchmarked locally using `llama3.1:8b` and `nomic-embed-text` on Apple Silicon (raw artifacts in `results/`):

### 1. RAG Benchmark (MS MARCO Subset — 50 queries, 406 passages)
* **Retrieval (Sentence + Hybrid RRF)**: Recall@1: `0.25` | Recall@3: `0.62` | Recall@5: `0.91` | MRR@10: `0.51`
* **Grounded Answer Quality**: Lexical Faithfulness: `0.89` | LLM-as-a-Judge Faithfulness: `0.94`
* **Success Rate**: 50/50 answered without errors; 12% appropriate abstention on insufficient context.

### 2. Website Builder Benchmark (3 Scenarios, 16 Edits)
* **Generation**: 3/3 valid on first attempt / repair loop.
* **Edit Correctness**: 79% applied accurately (100% preservation of untouched sections).
* **State Invariants**: 100% edit history integrity across consecutive modifications.

---

## Testing

```bash
pytest -q
```
**58 tests pass offline** in under 1 second without external network or API keys:
* `test_providers.py` (22 tests): HTTP transport, retries, runtime adaptation, error typing, key sanitization.
* `test_builder.py` (18 tests): Schema validation, patch atomicity, undo/versioning, conflict detection (409).
* `test_rag.py` (18 tests): Chunkers, BM25/dense/hybrid retrievers, grading fallback, query rewrite loop.
