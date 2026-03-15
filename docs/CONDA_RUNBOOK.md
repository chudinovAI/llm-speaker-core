# Conda Runbook

## Setup
```bash
git pull
conda create -n llm-speaker python=3.11 -y
conda activate llm-speaker
uv sync
```

## Ollama
```bash
ollama serve
ollama run qwen3.5:2b --think=false "Привет, ответь кратко"
```

## RAG indexing
```bash
uv run llm-build-index --nightly-reindex --raw suai_facts.txt --input data/cleaned_corpus.jsonl --report data/cleaning_report.json --output data/rag_index.json --min-words 16
```

## Voice stack
```bash
uv run llm-voice-stack --list-devices
uv run llm-voice-stack --mode direct --asr-device 0 --tts-play --wake-word "привет гуап" --no-speaker-verify
```

## Outputs
- `runtime/asr_output.txt`
- `runtime/asr_llm_output.jsonl`
- `runtime/speaker_output.txt`
- `runtime/voice_events.jsonl`
- `runtime/tts/*.wav`

