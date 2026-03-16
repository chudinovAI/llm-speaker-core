from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any

try:
    import fitz  # type: ignore[import-not-found]
    import pytesseract
    from PIL import Image
except Exception:  # noqa: BLE001
    fitz = None  # type: ignore[assignment]
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None and fitz is not None and pytesseract is not None and Image is not None


def ocr_pdf(path: Path, max_pages: int = 8) -> str:
    if not tesseract_available():
        return ""
    doc = fitz.open(path)
    pages: list[str] = []
    for page in doc[:max_pages]:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, lang="rus+eng")
        if text.strip():
            pages.append(text.strip())
    return "\n".join(pages).strip()
