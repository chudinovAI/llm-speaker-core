git pull
conda create -n llm-speaker python=3.11 -y
conda activate llm-speaker

uv sync
ollama serve
ollama run qwen3.5:2b --think=false "Привет, ответь кратко"

# RAG-indexing

uv run llm-build-index --nightly-reindex --raw suai_facts.txt --input data/cleaned_corpus.jsonl --report data/cleaning_report.json --output data/rag_index.json --min-words 16

---

uv run llm-voice-stack --list-devices
uv run llm-voice-stack --asr-device 1 --tts-play --wake-word "привет гуап" --no-speaker-verify
