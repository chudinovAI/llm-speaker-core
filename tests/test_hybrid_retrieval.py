from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_speaker_core.retrieval.lexical import LexicalIndex
from llm_speaker_core.retrieval.schemas import ChunkRecord, IndexManifest
from llm_speaker_core.retrieval.service import HybridRetrievalService


def _chunk(
    chunk_id: str,
    text: str,
    source: str,
    section: str,
    *,
    is_archived: bool = False,
    quality_flags: list[str] | None = None,
    quality_score: float = 1.0,
    title: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        source_url=source,
        canonical_url=source,
        source_type="web",
        title=title or section,
        section=section,
        published_at=None,
        language="ru",
        content_hash=chunk_id,
        chunk_index=0,
        token_count=len(text.split()),
        char_count=len(text),
        overlap_from_prev=0,
        ocr_used=False,
        quality_score=quality_score,
        text=text,
        section_path=[section],
        metadata=metadata or {},
        quality_flags=quality_flags or [],
        source=source,
        is_archived=is_archived,
        is_low_signal=False,
    )


def test_hybrid_search_prefers_relevant_official_source() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Поступление в ГУАП: абитуриенты подают документы через официальный раздел для абитуриентов.",
            "https://guap.ru/abitur",
            "abitur",
        ),
        _chunk(
            "doc2:0",
            "Контакты университета и адреса корпусов опубликованы в разделе сведений.",
            "https://guap.ru/sveden/common",
            "sveden",
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v3",
        corpus_checksum="abc",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-03-15T00:00:00Z",
        doc_count=2,
        chunk_count=2,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Как поступить в ГУАП?", top_k=2)

    assert hits
    assert hits[0]["source"] == "https://guap.ru/abitur"


def test_hybrid_loads_from_manifest_without_dense(tmp_path: Path) -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Стоимость обучения зависит от программы и формы обучения.",
            "https://guap.ru/eif/price",
            "eif",
            is_archived=False,
        )
    ]
    lexical = LexicalIndex.build(chunks)
    lexical_path = tmp_path / "bm25.json"
    lexical.save(lexical_path)
    manifest: dict[str, Any] = {
        "version": "hybrid-rag-v3",
        "corpus_checksum": "checksum",
        "lexical_path": lexical_path.name,
        "dense_path": None,
        "reranker_model": "",
        "embedding_model": "",
        "built_at": "2026-03-15T00:00:00Z",
        "doc_count": 1,
        "chunk_count": 1,
        "metadata": {},
    }
    manifest_path = tmp_path / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    service = HybridRetrievalService.load(manifest_path)
    evidence = service.build_evidence_pack("Сколько стоит обучение в ГУАП?", top_k=1)

    assert evidence.hits
    assert evidence.hits[0].source == "https://guap.ru/eif/price"
    assert evidence.grounding_score > 0


def test_hybrid_search_penalizes_faq_and_prefers_structured_contacts_page() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Вопросы и ответы о поступлении в ГУАП. Общая справочная информация для абитуриентов.",
            "https://guap.ru/faq",
            "faq",
            quality_flags=["is_low_priority_source"],
            quality_score=0.7,
            title="Вопросы и ответы",
        ),
        _chunk(
            "doc2:0",
            "Приемная комиссия ГУАП: контакты, телефон, почта и режим работы.",
            "https://priem.guap.ru/contacts",
            "contacts",
            title="Контакты приемной комиссии",
        ),
        _chunk(
            "doc3:0",
            "Правила приема ГУАП. Общие положения и сроки подачи документов.",
            "https://priem.guap.ru/bach/rules",
            "rules",
            title="Правила приема",
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v3",
        corpus_checksum="contacts",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-03-17T00:00:00Z",
        doc_count=3,
        chunk_count=3,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Как связаться с приемной комиссией ГУАП?", top_k=3)

    assert hits
    assert hits[0]["source"] == "https://priem.guap.ru/contacts"
    assert hits[1]["source"] != "https://guap.ru/faq"


def test_hybrid_search_prefers_calc_for_directions_query() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Калькулятор направлений приема в ГУАП. Подбор программ бакалавриата и магистратуры.",
            "https://priem.guap.ru/calc",
            "admission",
            title="Калькулятор направлений",
            metadata={"page_type": "catalog", "source_facets": ["admission", "admission_directions"]},
        ),
        _chunk(
            "doc2:0",
            "Правила приема в бакалавриат и специалитет ГУАП.",
            "https://priem.guap.ru/bach/rules",
            "admission",
            title="Правила приема",
            metadata={"page_type": "policy", "source_facets": ["admission", "admission_bach"]},
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v3",
        corpus_checksum="directions",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-03-18T00:00:00Z",
        doc_count=2,
        chunk_count=2,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Какие направления есть в ГУАП?", top_k=2)

    assert hits
    assert hits[0]["source"] == "https://priem.guap.ru/calc"


def test_hybrid_search_prefers_dates_for_deadline_query() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Сроки проведения приема в ГУАП. Даты подачи документов и зачисления.",
            "https://priem.guap.ru/bach/dates",
            "admission",
            title="Сроки проведения приема",
            metadata={"page_type": "schedule", "source_facets": ["admission", "admission_dates", "admission_bach"]},
        ),
        _chunk(
            "doc2:0",
            "Правила приема в бакалавриат и специалитет ГУАП.",
            "https://priem.guap.ru/bach/rules",
            "admission",
            title="Правила приема",
            metadata={"page_type": "policy", "source_facets": ["admission", "admission_bach"]},
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v3",
        corpus_checksum="dates",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-03-18T00:00:00Z",
        doc_count=2,
        chunk_count=2,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Какие сроки приема в ГУАП?", top_k=2)

    assert hits
    assert hits[0]["source"] == "https://priem.guap.ru/bach/dates"


def test_hybrid_search_prefers_student_unions_over_generic_clubs() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Студенческое самоуправление. Профком студентов и аспирантов ГУАП.",
            "https://guap.ru/studlife/pposa",
            "studlife",
            title="Профком студентов и аспирантов",
            metadata={"page_type": "organization", "source_facets": ["student_life", "student_unions"]},
        ),
        _chunk(
            "doc2:0",
            "Совет старост ГУАП. Студенческое самоуправление.",
            "https://guap.ru/studlife/starsovet",
            "studlife",
            title="Совет старост ГУАП",
            metadata={"page_type": "organization", "source_facets": ["student_life", "student_unions"]},
        ),
        _chunk(
            "doc3:0",
            "Киберспортивный клуб ГУАП. Турниры и тренировки.",
            "https://guap.ru/studlife/cyber",
            "studlife",
            title="Киберспортивный клуб",
            metadata={"page_type": "profile", "source_facets": ["student_life"]},
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v3",
        corpus_checksum="unions",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-03-18T00:00:00Z",
        doc_count=3,
        chunk_count=3,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Есть ли в ГУАП студенческие объединения?", top_k=3)

    assert hits
    assert hits[0]["source"] in {
        "https://guap.ru/studlife/pposa",
        "https://guap.ru/studlife/starsovet",
    }
    assert "https://guap.ru/studlife/cyber" not in {hit["source"] for hit in hits[:2]}


def test_hybrid_search_prefers_payment_pages_over_price_for_payment_query() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Оплата обучения и проживания в общежитиях. Реквизиты и порядок оплаты.",
            "https://guap.ru/eif/pay",
            "eif",
            title="Оплата обучения и проживания",
            metadata={"page_type": "detail", "source_facets": ["tuition", "tuition_payment"]},
        ),
        _chunk(
            "doc2:0",
            "Приказы о стоимости обучения. Цены и тарифы по программам.",
            "https://guap.ru/eif/price",
            "eif",
            title="Приказы о стоимости обучения",
            metadata={"page_type": "detail", "source_facets": ["tuition", "tuition_price"]},
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v3",
        corpus_checksum="payment",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-03-18T00:00:00Z",
        doc_count=2,
        chunk_count=2,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Как оплатить обучение в ГУАП?", top_k=2)

    assert hits
    assert hits[0]["source"] == "https://guap.ru/eif/pay"


def test_hybrid_search_prefers_specific_dorm_page_for_specific_dorm_query() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Общежитие 2 ГУАП. Адрес и информация о проживании.",
            "https://guap.ru/dom/2",
            "dom",
            title="Общежитие 2",
            metadata={"page_type": "facilities", "source_facets": ["dorm", "location"]},
        ),
        _chunk(
            "doc2:0",
            "Материально-техническое обеспечение и общежития ГУАП.",
            "https://guap.ru/sveden/objects",
            "sveden",
            title="Объекты и общежития",
            metadata={"page_type": "facilities", "source_facets": ["dorm", "location"]},
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v3",
        corpus_checksum="dorm-dom2",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-03-19T00:00:00Z",
        doc_count=2,
        chunk_count=2,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Где находится общежитие 2 ГУАП?", top_k=2)

    assert hits
    assert hits[0]["source"] == "https://guap.ru/dom/2"


def test_hybrid_search_uses_numeric_bonus_for_dorm_number() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Общежитие №2 ГУАП. Адрес, транспорт и администрация.",
            "https://guap.ru/dom/2",
            "dom",
            title="Общежитие №2",
            metadata={"page_type": "facilities", "source_facets": ["dorm", "location"]},
        ),
        _chunk(
            "doc2:0",
            "Общежития ГУАП. Общие сведения об объектах.",
            "https://guap.ru/sveden/objects",
            "sveden",
            title="Общежития",
            metadata={"page_type": "facilities", "source_facets": ["dorm", "location"]},
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v3",
        corpus_checksum="dorm-number",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-03-19T00:00:00Z",
        doc_count=2,
        chunk_count=2,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Где находится общежитие 2 ГУАП?", top_k=2)

    assert hits
    assert hits[0]["source"] == "https://guap.ru/dom/2"


def test_hybrid_search_prefers_org_schedule_page_for_operational_query() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Научная статья про расписание вычислительных задач и работу отделов в распределенной сети.",
            "https://guap.ru/science/projects",
            "science",
            title="Научные проекты",
            metadata={"page_type": "detail", "source_facets": []},
        ),
        _chunk(
            "doc2:0",
            "Отдел кадров ГУАП. Телефон, кабинет и режим работы: понедельник-пятница с 10 до 18.",
            "https://guap.ru/struct/hr",
            "struct",
            title="Отдел кадров",
            metadata={
                "page_type": "contacts",
                "source_facets": ["org_unit", "org_contacts", "org_schedule", "contacts"],
            },
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v3",
        corpus_checksum="org-schedule",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-04-07T00:00:00Z",
        doc_count=2,
        chunk_count=2,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Какой режим работы отдела кадров ГУАП?", top_k=2)

    assert hits
    assert hits[0]["source"] == "https://guap.ru/struct/hr"
