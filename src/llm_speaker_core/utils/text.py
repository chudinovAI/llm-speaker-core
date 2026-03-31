from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")
RU_STOPWORDS = {
    "и",
    "в",
    "на",
    "по",
    "с",
    "к",
    "о",
    "об",
    "у",
    "за",
    "из",
    "для",
    "что",
    "как",
    "какие",
    "какой",
    "где",
    "это",
    "есть",
    "ли",
    "или",
}
QUERY_EXPANSIONS = {
    "поступ": ("абитуриент", "прием", "приём", "баллы", "направления"),
    "направлен": ("специальности", "программы", "бакалавриат", "магистратура"),
    "специальн": ("направления", "программы", "бакалавриат", "магистратура"),
    "программ": ("направления", "специальности", "бакалавриат", "магистратура"),
    "приемн": ("приемная", "приёмная", "комиссия", "контакты", "поступление"),
    "сведен": ("официальные", "документы", "лицензия", "аккредитация"),
    "официаль": ("сведения", "документы", "лицензия", "аккредитация"),
    "общежит": ("проживание", "заселение", "кампус"),
    "стипенд": ("выплаты", "поддержка", "социальная"),
    "стои": ("стоимость", "цена", "оплата", "платное", "договор"),
    "скольк": ("стоимость", "цена", "оплата"),
    "стоим": ("цена", "обучение", "оплата", "платное", "договор"),
    "оплат": ("оплата", "стоимость", "цена", "обучение", "договор"),
    "цена": ("стоимость", "оплата", "обучение", "договор"),
    "контакт": ("приемная", "комиссия", "телефон", "почта", "email"),
    "связ": ("контакты", "телефон", "почта", "email"),
    "срок": ("даты", "прием", "приём"),
    "дат": ("сроки", "прием", "приём"),
    "бюдж": ("бюджетные", "места", "план"),
    "мест": ("бюджет", "план", "прием"),
    "объедин": ("самоуправление", "профком", "старостат"),
    "самоуправ": ("объединения", "профком", "старостат"),
    "врмп": ("воспитательная", "молодежная", "политика", "точка"),
    "магистр": ("магистратура", "маг"),
    "гуап": ("университет", "аэрокосмического", "приборостроения"),
}
INTENT_HINTS: dict[str, tuple[str, ...]] = {
    "admission": (
        "поступ",
        "абитуриент",
        "прием",
        "приём",
        "баллы",
        "комисси",
        "направлен",
        "специальн",
        "программ",
        "бакалавр",
        "магистр",
    ),
    "contacts": ("контакт", "связ", "телефон", "почт", "email", "приемн"),
    "location": ("где", "адрес", "корпус", "кампус", "наход"),
    "student_life": ("студен", "актив", "круж", "клуб", "театр", "спорт"),
    "official_info": ("сведен", "официальн", "документ", "лиценз", "аккредит"),
    "tuition": (
        "стоим",
        "стои",
        "оплат",
        "цена",
        "обучен",
        "платн",
        "договор",
        "контракт",
    ),
}

FACET_HINTS: dict[str, tuple[str, ...]] = {
    "admission_directions": ("направлен", "специальн", "калькулятор", "программ", "профил"),
    "admission_dates": ("срок", "дат"),
    "admission_contacts": ("приемн", "приёмн", "комисси", "контакт"),
    "admission_budget": ("бюдж", "мест"),
    "admission_mag": ("магистр", "магистрат"),
    "tuition_price": ("стоим", "стои", "цена", "скольк"),
    "tuition_payment": ("оплат", "договор", "реквиз"),
    "student_unions": ("объедин", "самоуправ", "профком", "ппоса", "старост"),
    "dorm": ("общежит", "проживан"),
    "vrmp": ("врмп", "точка", "кипения"),
}


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in RU_STOPWORDS and len(t) > 1]


def expand_query(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        for stem, extra in QUERY_EXPANSIONS.items():
            if token.startswith(stem):
                expanded.extend(extra)
    return expanded


def detect_facets(text: str) -> list[str]:
    low = text.lower()
    expanded_tokens = expand_query(tokenize(text))
    facets: list[str] = []
    for facet, stems in FACET_HINTS.items():
        if any(token.startswith(stem) for token in expanded_tokens for stem in stems):
            facets.append(facet)
            continue
        if any(stem in low for stem in stems):
            facets.append(facet)

    if "admission_mag" in facets and "admission_directions" not in facets:
        facets.append("admission_directions")
    if "admission_contacts" in facets and "contacts" not in facets:
        facets.append("contacts")
    if "dorm" in facets and "location" not in facets:
        facets.append("location")
    if "tuition_price" in facets and "tuition_payment" in facets:
        if any(token.startswith(("скольк", "стоим", "стои", "цена")) for token in expanded_tokens):
            facets = [facet for facet in facets if facet != "tuition_payment"] + ["tuition_price"]
    return sorted(set(facets))
