#!/usr/bin/env bash
# Start PromptSite on a local model with one command (or double-click start.command on macOS).
#
#   ./run.sh                      # llama3.1:8b (8 billion parameters), http://localhost:8000
#   MODEL=glm4:9b ./run.sh        # any other Ollama model
#   EMBED_MODEL=... ./run.sh      # embedding model for the RAG page (default nomic-embed-text)
#   PORT=8080 ./run.sh
#   OPEN_BROWSER=0 ./run.sh       # do not open the browser
#
# It creates .venv and installs the dependencies if needed, starts Ollama if it is
# not running, downloads the models if they are missing, then serves the app:
# the website builder at / and the RAG page at /rag.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-llama3.1:8b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
PORT="${PORT:-8000}"
OLLAMA_URL="http://localhost:11434"

if ! command -v ollama >/dev/null; then
  if command -v brew >/dev/null; then
    echo "Installing Ollama with Homebrew"
    brew install ollama
  else
    echo "Ollama is not installed. Install it from https://ollama.com/download and run this again." >&2; exit 1
  fi
fi

if [ ! -x .venv/bin/python ]; then
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
      PYTHON="$candidate"; break
    fi
  done
  [ -n "${PYTHON:-}" ] || { echo "Python 3.10 or newer is required." >&2; exit 1; }
  echo "Creating .venv with $PYTHON"
  "$PYTHON" -m venv .venv
fi
.venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt

if ! curl -sf "$OLLAMA_URL/api/version" >/dev/null; then
  echo "Starting Ollama"
  # 8192 tokens: the edit prompt carries the schema and the whole site JSON.
  OLLAMA_CONTEXT_LENGTH=8192 ollama serve >/dev/null 2>&1 &
  for _ in $(seq 1 30); do curl -sf "$OLLAMA_URL/api/version" >/dev/null && break; sleep 1; done
  curl -sf "$OLLAMA_URL/api/version" >/dev/null || { echo "Ollama did not start." >&2; exit 1; }
fi

for model in "$MODEL" "$EMBED_MODEL"; do
  if ! ollama show "$model" >/dev/null 2>&1; then
    echo "Downloading $model (first run only)"
    ollama pull "$model"
  fi
done
echo "Model: $MODEL ($(ollama show "$MODEL" | awk '$1 == "parameters" { print $2; exit }') parameters)"

# Exported variables take precedence over .env, so this run always uses the local model.
export LLM_PROVIDER=llama LLM_MODEL="$MODEL" LLM_BASE_URL="$OLLAMA_URL/v1" LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-300}"
export EMBEDDING_PROVIDER=llama EMBEDDING_MODEL="$EMBED_MODEL" EMBEDDING_BASE_URL="$OLLAMA_URL/v1"
echo "PromptSite: http://localhost:$PORT  (Website Builder at /, Document Chat at /rag). Stop with Ctrl+C."
if [ "${OPEN_BROWSER:-1}" = 1 ]; then
  opener="$(command -v open || command -v xdg-open || true)"     # macOS, Linux
  if [ -n "$opener" ]; then
    ( for _ in $(seq 1 60); do curl -sf "http://localhost:$PORT/api/health" >/dev/null && { "$opener" "http://localhost:$PORT"; break; }; sleep 1; done ) &
  fi
fi
exec .venv/bin/uvicorn server:create_app --factory --port "$PORT"
