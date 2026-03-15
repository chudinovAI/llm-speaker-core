# llm-speaker-core

Текущий стек:
- voice runtime: `ASR -> LLM/RAG -> TTS`
- HTTP API для отладки и Bruno
- текущий RAG сохранен как baseline до полного rewrite

## Main Commands
```bash
uv sync
uv run llm-build-index --nightly-reindex --raw suai_facts.txt --input data/cleaned_corpus.jsonl --report data/cleaning_report.json --output data/rag_index.json --min-words 16
uv run llm-voice-stack --mode direct --asr-device 0 --tts-play --wake-word "привет гуап" --no-speaker-verify
uv run llm-api
uv run pytest -q
uv run mypy .
```

## Docs
- [Runbook](/Users/andreychudinov/Documents/Projects/llm-speaker-core/docs/CONDA_RUNBOOK.md)
- [Architecture](/Users/andreychudinov/Documents/Projects/llm-speaker-core/docs/ARCHITECTURE.md)
