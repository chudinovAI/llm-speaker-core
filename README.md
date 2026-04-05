## Preinstall

- uv
- ollama
- git + git-lfs

## Setup

```bash
git pull
git lfs pull
uv sync
```

## Step 1. Download embedding model

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ai-sage/Giga-Embeddings-instruct')"
```

## Step 2. Build RAG index

```bash
uv run llm-build-hybrid-index
```

## Step 3. Start Ollama

```bash
ollama serve
ollama run qwen3.5:2b --think=false "Привет"
```

## Step 4. Run voice stack

```bash
uv run llm-voice-stack --list-devices

uv run llm-voice-stack \
 --asr-device 0 \
 --tts-play \
 --wake-word "привет гуап" \
 --no-speaker-verify
```

## Retrieval modes

- `--retrieval-mode fast` (default) — BM25 only, no GPU needed
- `--retrieval-mode full` — BM25 + dense (Giga-Embeddings-instruct) + reranker

## Validation

```bash
uv run mypy .
uv run pytest -q
```

## Data pipeline

```
1. Crawl        →  uv run llm-crawl-cloudflare --account-id ... --api-token ...
2. Normalize    →  (runs automatically inside llm-build-hybrid-index)
3. Build index  →  uv run llm-build-hybrid-index
4. Evaluate     →  uv run llm-eval-hybrid-rag
```
