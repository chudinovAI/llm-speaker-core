from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from llm_speaker_core.indexer import _is_archived_row, _row_intent_tags, build_multi_index
from llm_speaker_core.rag import LexicalRAG, tokenize


def _write_single_index(path: Path, docs: list[dict]) -> None:
    doc_tfs: list[dict[str, int]] = []
    doc_lens: list[int] = []
    df: Counter[str] = Counter()

    for doc in docs:
        tf = Counter(tokenize(str(doc["text"])))
        doc_tfs.append(dict(tf))
        doc_lens.append(sum(tf.values()))
        for token in tf.keys():
            df[token] += 1

    payload = {
        "docs": docs,
        "doc_tfs": doc_tfs,
        "doc_lens": doc_lens,
        "doc_freqs": dict(df),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_intent_tagger_filters_noise_for_tuition() -> None:
    noisy_doc = {
        "source_type": "doc",
        "source": "17e8a3874c_sbor70_4.pdf",
        "title": "Сборник материалов",
        "text": "Общечеловеческая ценность университета и культурные проекты для студентов.",
    }
    tags = _row_intent_tags(noisy_doc)
    assert "tuition" not in tags


def test_intent_tagger_marks_real_tuition_page() -> None:
    tuition_page = {
        "source_type": "web",
        "source": "https://guap.ru/eif/pay",
        "title": "Оплата обучения",
        "text": "Стоимость обучения, оплата обучения и формы договора для студентов ГУАП.",
    }
    tags = _row_intent_tags(tuition_page)
    assert "tuition" in tags


def test_rag_prefers_tuition_source_for_stoit_query(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    docs = [
        {
            "id": "web:1",
            "source": "https://guap.ru/sveden/common",
            "source_type": "web",
            "title": "Основные сведения",
            "text": "Федеральное образовательное учреждение в Санкт-Петербурге.",
        },
        {
            "id": "web:2",
            "source": "https://guap.ru/eif/pay",
            "source_type": "web",
            "title": "Оплата обучения",
            "text": (
                "Стоимость обучения по программам бакалавриата. "
                "Оплата обучения и формы договора доступны абитуриентам."
            ),
        },
        {
            "id": "doc:3",
            "source": "94314757d0_2021_9.pdf",
            "source_type": "doc",
            "title": "Путешествие",
            "text": "Студенчество в цифрах и ценность обучения в университете.",
        },
        {
            "id": "web:4",
            "source": "https://guap.ru/pubs/123",
            "source_type": "web",
            "title": "Новость",
            "text": "Университет провел культурное мероприятие для студентов.",
        },
    ]
    _write_single_index(index_path, docs)

    rag = LexicalRAG(index_path)
    hits = rag.search("Сколько стоит обучение в ГУАП?", top_k=3)
    assert hits
    assert hits[0]["source"] == "https://guap.ru/eif/pay"


def test_rag_prefers_admission_contact_page(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    docs = [
        {
            "id": "web:1",
            "source": "https://guap.ru/m/prioritet2030",
            "source_type": "web",
            "title": "Информация",
            "text": "Комиссия Министерства науки рассмотрела программу развития.",
        },
        {
            "id": "doc:2",
            "source": "cbdbeab337_pravila_2.pdf",
            "source_type": "doc",
            "title": "Правила подготовки материалов",
            "text": "Экспертная комиссия дает заключение о публикации материалов.",
        },
        {
            "id": "web:3",
            "source": "https://guap.ru/priem",
            "source_type": "web",
            "title": "Приемная комиссия",
            "text": "Контакты приемной комиссии: телефон, почта и часы работы.",
        },
    ]
    _write_single_index(index_path, docs)

    rag = LexicalRAG(index_path)
    hits = rag.search("Как связаться с приемной комиссией ГУАП?", top_k=3)
    assert hits
    assert hits[0]["source"] == "https://guap.ru/priem"


def test_detect_intent_for_directions_query() -> None:
    rag = LexicalRAG(Path("dummy.json"))
    intents = rag.detect_intents("Какие направления есть в ГУАПе?")
    assert "admission" in intents


def test_archived_row_detection() -> None:
    old_row = {
        "source_type": "web",
        "source": "https://guap.ru/eif/price",
        "title": "Стоимость обучения",
        "date": "2014",
        "text": "Стоимость обучения в 2014/2015 учебном году.",
    }
    fresh_row = {
        "source_type": "web",
        "source": "https://guap.ru/eif/price",
        "title": "Стоимость обучения",
        "date": "2025",
        "text": "Стоимость обучения на 2025/2026 учебный год.",
    }
    assert _is_archived_row(old_row) is True
    assert _is_archived_row(fresh_row) is False


def test_build_multi_index_skips_archived_for_sensitive_intents(tmp_path: Path) -> None:
    rows = [
        {
            "id": "web:old",
            "source_type": "web",
            "source": "https://guap.ru/eif/price",
            "title": "Стоимость обучения",
            "date": "2014",
            "text": "Стоимость обучения в 2014/2015 учебном году.",
        },
        {
            "id": "web:new",
            "source_type": "web",
            "source": "https://guap.ru/eif/price",
            "title": "Стоимость обучения",
            "date": "2025",
            "text": "Стоимость обучения на 2025/2026 учебный год и формы договора.",
        },
    ]
    manifest = tmp_path / "rag_index.json"
    report = build_multi_index(rows, manifest)
    tuition_path = Path(report["indexes"]["tuition"]["path"])
    tuition_payload = json.loads(tuition_path.read_text(encoding="utf-8"))
    doc_ids = {doc["id"] for doc in tuition_payload["docs"]}
    assert "web:new" in doc_ids
    assert "web:old" not in doc_ids


def test_build_multi_index_manifest_uses_relative_paths(tmp_path: Path) -> None:
    rows = [
        {
            "id": "web:1",
            "source_type": "web",
            "source": "https://guap.ru/sveden/common",
            "title": "Основные сведения",
            "date": "2025",
            "text": "ГУАП в Санкт-Петербурге.",
        }
    ]
    manifest = tmp_path / "rag_index.json"
    build_multi_index(rows, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["format"] == "multi_v1"
    assert payload["indexes"]["general"] == "rag_index.general.json"


def test_rag_loads_manifest_with_stale_absolute_paths_from_local_directory(tmp_path: Path) -> None:
    docs = [
        {
            "id": "web:1",
            "source": "https://guap.ru/sveden/common",
            "source_type": "web",
            "title": "Основные сведения",
            "text": "ГУАП находится в Санкт-Петербурге.",
        }
    ]
    sub_index = tmp_path / "rag_index.general.json"
    _write_single_index(sub_index, docs)
    manifest = tmp_path / "rag_index.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "multi_v1",
                "indexes": {
                    "general": "/Users/andreychudinov/Documents/Projects/llm-speaker-core/data/rag_index.general.json"
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rag = LexicalRAG(manifest)
    hits = rag.search("Где находится ГУАП?", top_k=1)
    assert hits
    assert hits[0]["source"] == "https://guap.ru/sveden/common"
