# Conda Runbook (Windows + uv + external Ollama)

## 0) Получить код проекта
```bash
git clone <repo_url>
cd <repo_dir>
```

## 1) Prerequisites
- Установлен Miniconda/Anaconda.
- Установлен `uv` (глобально).
- Установлен Ollama (отдельно от conda).
- Python 3.11 будет внутри conda env.

## 2) Создать и активировать conda-окружение
```bash
conda create -n llm-speaker python=3.11 -y
conda activate llm-speaker
```

## 3) Установить зависимости проекта через uv
Из корня репозитория:
```bash
uv sync
```

Если нужно явно:
```bash
uv pip install -e .
```

## 4) Проверить локальный Ollama (вне conda)
Откройте отдельный терминал (можно обычный PowerShell), запустите:
```bash
ollama serve
```

В другом терминале (можно внутри conda) подтяните модель:
```bash
ollama pull qwen2.5:3b
```

Проверка:
```bash
ollama run qwen2.5:3b "Привет, ответь кратко"
```

## 5) Подготовить RAG-данные (очистка + индекс)
Из корня проекта:
```bash
uv run llm-build-index --nightly-reindex --raw suai_facts.txt --input data/cleaned_corpus.jsonl --report data/cleaning_report.json --output data/rag_index.json
```

Проверить retrieval:
```bash
uv run llm-eval-rag --index data/rag_index.json --out data/rag_eval_report.json
```

## 6) Запуск API
```bash
uv run llm-api
```

Сервис поднимется на `http://127.0.0.1:8000`.

## 7) Проверка health/metrics/query
Health:
```bash
curl http://127.0.0.1:8000/health
```

Metrics:
```bash
curl http://127.0.0.1:8000/metrics
```

Query:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text":"Расскажи про ГУАП","session_id":"demo-1"}'
```

## 8) Ежедневный ручной reindex (nightly)
```bash
conda activate llm-speaker
uv run llm-build-index --nightly-reindex
```

## 9) Запуск тестов
```bash
uv run pytest -q
```

## 10) Troubleshooting
1. `Connection refused` к Ollama:
- Проверьте, что `ollama serve` запущен.
- Проверьте URL `http://127.0.0.1:11434`.

2. `RAG index not found`:
- Выполните nightly reindex команду из шага 4.

3. Долгий ответ LLM:
- Проверьте загрузку GPU.
- Уменьшите `num_predict`/лимиты слов в сервисе.

4. Пустой `speaker_text`:
- Это штатный fallback, когда speaker-компрессия не прошла.
- `display_text` в этом случае должен оставаться валидным.
