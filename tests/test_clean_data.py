from pathlib import Path

from llm_speaker_core.clean_data import run_cleaning


def test_run_cleaning_produces_jsonl_and_report(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text(
        """--- Источник: https://guap.ru/page1 ---
Главная
Подробнее
Это важная информация о поступлении в ГУАП в 2026 году.
--- Источник Документ: policy.pdf ---
Положение о подразделении
Положение о подразделении
Документ описывает правила обучения и стипендий для студентов.
""",
        encoding="utf-8",
    )

    out = tmp_path / "cleaned.jsonl"
    report_path = tmp_path / "report.json"

    rows, report = run_cleaning(raw, out, report_path, min_words=5)

    assert rows > 0
    assert out.exists()
    assert report_path.exists()
    assert report["sections_total"] == 2
    assert report["sections_web"] == 1
    assert report["sections_doc"] == 1
