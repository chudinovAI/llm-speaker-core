## Preinstall

- miniconda
- ollama
- git + git-lfs

## Setup

git pull
conda create -n llm-speaker python=3.11 -y
conda activate llm-speaker
uv sync

## Ollama

ollama serve
ollama run qwen3.5:2b --think=false "Привет"

## Voice

uv run llm-voice-stack --list-devices

uv run llm-voice-stack \
 --asr-device 0 \
 --tts-play \
 --wake-word "привет гуап" \
 --no-speaker-verify

## Validation

uv run mypy .
uv run pytest -q
