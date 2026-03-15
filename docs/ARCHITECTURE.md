# Architecture

## Layers
- `app/` - bootstrap, API wiring, health.
- `voice/` - ASR, TTS, stack orchestration, voice events, session state.
- `llm_runtime/` - LLM client implementation.
- `rag/indexer/clean_data/...` - current retrieval pipeline, preserved as-is for now.

## Voice runtime
```text
ASR -> transcript_final -> bridge -> LLMService -> display/speaker text -> TTS
```

## Canonical entrypoints
- `uv run llm-api`
- `uv run llm-voice-stack`
- `uv run llm-build-index`
- `uv run llm-eval-rag`

