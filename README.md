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
ollama run qwen3.5:9b --think=false "Привет"
```

## Step 4. Run voice stack

```bash
uv run llm-voice-stack --list-devices

uv run llm-voice-stack \
 --asr-device 0 \
 --wake-word "привет коробка" \
 --no-speaker-verify
```

Speaker playback is enabled by default; the female voice `kseniya` is selected by default. To record WAV without playback: `--no-tts-play`.

In the `runtime/` directory (or `--runtime-dir`): the full response text for the UI is in **`display_output.txt`** (text blocks separated by `---`), speech output is in **`speaker_output.txt`**, and the full log is in **`asr_llm_output.jsonl`**.

## Retrieval modes

- `--retrieval-mode fast` (default) — BM25 only, no GPU needed
- `--retrieval-mode full` — BM25 + dense (Giga-Embeddings-instruct) + reranker

## Validation

```bash
uv run mypy .
uv run pytest -q
```
