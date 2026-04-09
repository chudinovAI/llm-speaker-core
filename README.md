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

Звук в динамики включён по умолчанию; женский голос `kseniya` выбран по умолчанию. Только запись WAV без проигрывания: `--no-tts-play`.

В каталоге `runtime/` (или `--runtime-dir`): полный текст ответа для UI — **`display_output.txt`** (блоки текста, разделитель `---`), озвучка — **`speaker_output.txt`**, полный лог — **`asr_llm_output.jsonl`**.

## Retrieval modes

- `--retrieval-mode fast` (default) — BM25 only, no GPU needed
- `--retrieval-mode full` — BM25 + dense (Giga-Embeddings-instruct) + reranker

## Validation

```bash
uv run mypy .
uv run pytest -q
```
