# llm-speaker-core

LLM-core for smart speaker backend (text->text):
- `display_text` for screen (Markdown-lite, <=120 words)
- `speaker_text` for TTS (1 sentence, <=20 words)

## Modules
- `llm_speaker_core.clean_data` - cleaning raw corpus into JSONL.
- `llm_speaker_core.indexer` - lexical RAG index builder and nightly reindex.
- `llm_speaker_core.evaluate_rag` - retrieval validation report.
- `llm_speaker_core.api` - FastAPI service (`/health`, `/metrics`, `/query`).

## Main Commands
```bash
uv run llm-build-index --nightly-reindex
uv run llm-eval-rag
uv run llm-api
uv run pytest -q
```

Detailed environment/setup guide: [CONDA_RUNBOOK.md](/Users/andreychudinov/Documents/Projects/llm-speaker-core/CONDA_RUNBOOK.md)
Implementation plan snapshot: [IMPLEMENTATION_PLAN.md](/Users/andreychudinov/Documents/Projects/llm-speaker-core/IMPLEMENTATION_PLAN.md)
