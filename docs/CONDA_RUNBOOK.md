# Conda Runbook (Windows + uv + external Ollama)

## 0) Получить код проекта
```bash
git clone https://github.com/chudinovAI/llm-speaker-core.git
cd llm-speaker-core
git checkout codex/local-speaker-compression
```

## 0.1) Что не лежит в git и что нужно перенести отдельно
Готовые RAG-данные и raw-корпус в репозиторий не включены. Чтобы на рабочей машине не пересобирать корпус с нуля, скопируйте с текущей машины:

- `suai_facts.txt`
- `data/cleaned_corpus.jsonl`
- `data/cleaning_report.json`
- `data/rag_index.json`
- `data/rag_index.admission.json`
- `data/rag_index.contacts.json`
- `data/rag_index.general.json`
- `data/rag_index.location.json`
- `data/rag_index.news.json`
- `data/rag_index.policies.json`
- `data/rag_index.student_life.json`
- `data/rag_index.tuition.json`

Минимум для запуска без reindex:
- все `data/rag_index*.json`

Нужно для повторной сборки индекса без повторного парсинга:
- `suai_facts.txt`
- `data/cleaned_corpus.jsonl`

Пример копирования через `scp` с Mac/Linux на рабочую машину:
```bash
scp suai_facts.txt user@WORK_PC:/path/to/llm-speaker-core/
scp data/cleaned_corpus.jsonl data/cleaning_report.json data/rag_index*.json user@WORK_PC:/path/to/llm-speaker-core/data/
```

Если `scp` неудобен, можно просто перенести эти файлы через флешку, облако или общий диск, сохранив те же пути в корне проекта и в папке `data/`.

## 1) Prerequisites
- Установлен Miniconda/Anaconda.
- Установлен `uv` (глобально).
- Установлен Ollama (отдельно от conda).
- Python 3.11 будет внутри conda env.

## 2) Создать и активировать conda-окружение
Если `conda activate` не работает:
```bash
conda init powershell
```

Закройте и откройте терминал заново, затем:
```bash
conda create -n llm-speaker python=3.11 -y
conda activate llm-speaker
```

Если conda просит принять Terms of Service:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
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

Если видите `address already in use`, значит Ollama уже запущен. Второй `ollama serve` не нужен.

В другом терминале (можно внутри conda) подтяните модель:
```bash
ollama pull qwen3.5:2b
```

Проверка:
```bash
ollama run qwen3.5:2b --think=false "Привет, ответь кратко"
```

## 5) Подготовить RAG-данные
### Вариант A: использовать уже готовые данные
Если вы уже перенесли `data/rag_index*.json`, этот шаг можно пропустить.

Быстрая проверка:
```bash
dir data
```

### Вариант B: пересобрать индекс локально из уже перенесенного raw/cleaned корпуса
Из корня проекта:
```bash
uv run llm-build-index --nightly-reindex --raw suai_facts.txt --input data/cleaned_corpus.jsonl --report data/cleaning_report.json --output data/rag_index.json --min-words 16
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
uv run llm-build-index --nightly-reindex --raw suai_facts.txt --input data/cleaned_corpus.jsonl --report data/cleaning_report.json --output data/rag_index.json --min-words 16
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
- Либо скопируйте `data/rag_index*.json` с основной машины.
- Либо выполните reindex команду из шага 5.

3. Долгий ответ LLM:
- Проверьте загрузку GPU.
- Уменьшите `num_predict`/лимиты слов в сервисе.

4. `uv sync` падает на Git-зависимости:
- Проверьте, что установлен `git`.
- Повторите `uv sync` после проверки доступа к GitHub.

5. Пустой `speaker_text`:
- Это штатный fallback, когда speaker-компрессия не прошла.
- `display_text` в этом случае должен оставаться валидным.

6. Нужно быстро поднять проект на новой машине без reindex:
- Сначала перенесите `data/rag_index*.json`.
- Затем выполните только шаги 0, 2, 3, 4 и 6.
