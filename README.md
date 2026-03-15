# llm-speaker-core

Голосовой стек для умной колонки:
- `display_text` для дисплея;
- `speaker_text` для TTS;
- RAG по корпусу ГУАП;
- локальный voice runtime: `ASR -> LLM/RAG -> TTS`.

## Structure
- `src/llm_speaker_core/` - ядро LLM, RAG, API.
- `src/llm_speaker_core/voice/` - ASR, bridge, TTS, unified stack runner.
- `docs/` - runbook и зафиксированный план.
- `data/` - cleaned corpus и RAG indexes.
- `runtime/` - временные файлы во время работы voice stack.
- `tests/` - тесты.

## Main Commands
```bash
uv run llm-build-index --nightly-reindex --raw suai_facts.txt --input data/cleaned_corpus.jsonl --report data/cleaning_report.json --output data/rag_index.json --min-words 16
uv run llm-eval-rag --index data/rag_index.json --out data/rag_eval_report.json
uv run llm-api
uv run llm-voice-stack --tts-play --wake-word "привет гуап" --no-speaker-verify
uv run pytest -q
```

## Voice Modules
- `llm_speaker_core.voice.asr` - GigaAM streaming ASR.
- `llm_speaker_core.voice.bridge` - ASR text -> direct LLM/API -> speaker text/TTS.
- `llm_speaker_core.voice.tts` - Silero TTS.
- `llm_speaker_core.voice.stack` - one-command runtime for the whole voice pipeline.

## Docs
- Runbook: [docs/CONDA_RUNBOOK.md](/Users/andreychudinov/Documents/Projects/llm-speaker-core/docs/CONDA_RUNBOOK.md)
- Plan snapshot: [docs/IMPLEMENTATION_PLAN.md](/Users/andreychudinov/Documents/Projects/llm-speaker-core/docs/IMPLEMENTATION_PLAN.md)
