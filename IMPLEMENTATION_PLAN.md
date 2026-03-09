# Full Implementation Plan (LLM Core + Data Cleaning + Conda Runbook)

## Summary
Реализуем LLM-ядро умной колонки (text->text) с двумя каналами выхода в одном ответе:
1. `display_text` (Markdown-lite, до 120 слов).
2. `speaker_text` (1 предложение, до 20 слов, ключевая суть для TTS).

Перед интеграцией API выполняется очистка и нормализация raw-корпуса RAG.
Стек: `uv`, `Python 3.11`, `Windows + conda`, `Ollama + Qwen2.5 3B`.

## Key Changes
1. Data Cleaning First
- Разделение данных на web/doc потоки.
- Удаление boilerplate (меню, футеры, контакты, повторяющиеся строки).
- Дедуп чанков и фильтр коротких фрагментов.
- Метаданные источника (`source`, `source_type`, `title`, `date`).
- Отчет качества очистки: объем до/после, retention, удаленные дубликаты и мусор.

2. RAG Indexing
- Индексация только cleaned corpus.
- Лексический retriever и базовая валидация retrieval на тестовых вопросах.
- Nightly reindex вручную повторяемой командой.

3. LLM API
- `POST /query`: вход `text`, `session_id`, опциональная история.
- Генерация `display_text` + `speaker_text`.
- Жесткие guardrails, русский-only, память 3-5 реплик.
- Fallback: если speaker-ветка падает, `speaker_text=""`, display не ломается.

4. Performance/Runtime
- Цель p95 <= 5 сек при 1 одновременном пользователе.
- Ограничение длины/токенов.
- Логи и счетчики: latency, fallback, empty speaker, rag hits.

5. Документация
- План в этом файле.
- Пошаговый запуск в `CONDA_RUNBOOK.md`.

## Public Interfaces
- Input DTO: `text: str`, `session_id: str`, `history?: list`.
- Output DTO:
  - `display_text: str`
  - `speaker_text: str`
  - `meta: { latency_ms, used_rag, fallback_used, limits_applied, rag_hits }`

## Test Plan
1. Контракт API (оба поля ответа, fallback-сценарий).
2. Лимиты и формат (`display<=120`, `speaker<=20`, speaker без markdown).
3. Guardrails и русскоязычный формат.
4. RAG retrieval на cleaned corpus.
5. Последовательная нагрузка и проверка SLA.

## Assumptions
- STT и TTS реализуются другими командами.
- `speaker_text` генерируется всегда.
- OCR для scanned PDF — следующий этап, не блокирует MVP.
