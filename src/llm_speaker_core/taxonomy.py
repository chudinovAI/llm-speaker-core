from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class IntentProfile:
    query_hints: tuple[str, ...] = ()
    fact_tokens: frozenset[str] = frozenset()
    trusted_source_hints: tuple[str, ...] = ()
    preferred_source_hints: tuple[str, ...] = ()
    allowed_source_hints: tuple[str, ...] = ()
    denied_source_hints: tuple[str, ...] = ()
    noisy_sentence_markers: tuple[str, ...] = ()
    query_boost_terms: tuple[str, ...] = ()
    display_summary: str = ""
    speaker_summary: str = ""
    min_evidence_for_freeform: float = 0.35
    min_fact_token_hits: int = 2
    priority: int = 0


QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
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
    "кадр": ("отдел", "персонал", "сотрудники"),
    "декан": ("деканат", "факультет"),
    "отдел": ("подразделение", "служба", "управление"),
    "подраздел": ("отдел", "служба", "управление"),
    "режим": ("график", "часы", "расписание"),
    "график": ("режим", "часы", "расписание"),
    "библиот": ("читальный", "абонемент", "книги"),
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
    "org_unit": ("отдел", "кадр", "декан", "деканат", "подраздел", "управлен", "служб", "библиот", "кафедр"),
    "org_contacts": ("контакт", "телефон", "почт", "email", "кабинет"),
    "org_schedule": ("режим", "график", "часы", "расписан"),
    "org_staff": ("сотрудник", "персонал", "руководител", "состав"),
}

ORGANIZATION_MARKERS = (
    "отдел",
    "управление",
    "служба",
    "центр",
    "деканат",
    "кафедра",
    "библиотек",
    "лаборатория",
    "приемная комиссия",
    "приёмная комиссия",
)
ORGANIZATION_PATH_MARKERS = (
    "/struct",
    "/department",
    "/dean",
    "/library",
    "/empbook",
    "/employees",
    "/managers",
)
ORGANIZATION_CONTACT_MARKERS = (
    "контакты",
    "телефон",
    "почта",
    "email",
    "e-mail",
    "@guap.ru",
    "адрес",
    "кабинет",
)
ORGANIZATION_SCHEDULE_MARKERS = (
    "режим работы",
    "график работы",
    "часы работы",
    "время работы",
    "расписание работы",
    "приемные часы",
    "приёмные часы",
)
ORGANIZATION_STAFF_MARKERS = (
    "сотрудники",
    "персонал",
    "руководитель",
    "начальник",
    "заведующ",
    "состав подразделения",
)

PATH_TOKEN_ALIASES = {
    "bach": ("бакалавриат", "специалитет", "поступление", "направления"),
    "mag": ("магистратура", "магистр", "программы"),
    "dates": ("сроки", "даты", "прием"),
    "contacts": ("контакты", "телефон", "адрес", "приемная", "комиссия"),
    "rules": ("правила", "прием", "поступление"),
    "pay": ("оплата", "стоимость", "обучение"),
    "pay_edu": ("оплата", "платные", "услуги", "обучение"),
    "paid_edu": ("оплата", "платные", "услуги", "обучение"),
    "price": ("стоимость", "цена", "обучение"),
    "common": ("сведения", "контакты", "адрес", "режим"),
    "objects": ("общежитие", "корпуса", "объекты", "адрес"),
    "budget": ("бюджет", "бюджетные", "места"),
    "grants": ("стипендия", "поддержка"),
    "studlife": ("студенты", "активности", "объединения", "клубы"),
    "vrmp": ("воспитательная", "молодежная", "политика"),
    "tochka": ("точка", "кипения"),
    "calc": ("направления", "калькулятор", "программы", "профили"),
    "plan": ("бюджет", "места", "прием"),
    "pposa": ("объединения", "профком", "самоуправление"),
    "starsovet": ("объединения", "старостат", "самоуправление"),
    "domsovet": ("общежитие", "объединения", "самоуправление"),
    "managers": ("руководство", "сотрудники", "контакты"),
    "employees": ("сотрудники", "персонал", "состав"),
    "struct": ("структура", "подразделение", "отдел", "управление"),
    "library": ("библиотека", "читальный", "абонемент"),
    "dean": ("деканат", "факультет"),
    "faq": ("вопросы", "ответы"),
}

FACET_RULES = {
    "admission_directions": {
        "path_bonus": {"/calc": 0.44, "/bach": 0.18, "/mag": 0.18},
        "path_penalty": {"/rules": 0.24, "/contacts": 0.16},
        "page_type_bonus": {"catalog": 0.34, "hub": 0.14},
        "page_type_penalty": {"policy": 0.22, "contacts": 0.12, "directory": 0.12},
        "source_facet_bonus": {"admission_directions": 0.44, "admission_bach": 0.14, "admission_mag": 0.14},
        "source_facet_penalty": {},
    },
    "admission_dates": {
        "path_bonus": {"/dates": 0.62},
        "path_penalty": {"/rules": 0.34, "/plan": 0.16},
        "page_type_bonus": {"schedule": 0.46},
        "page_type_penalty": {"policy": 0.22, "catalog": 0.12, "plan": 0.14},
        "source_facet_bonus": {"admission_dates": 0.54},
        "source_facet_penalty": {},
    },
    "admission_contacts": {
        "path_bonus": {"/contacts": 0.62, "priem.guap.ru/contacts": 0.2},
        "path_penalty": {"/sveden/managers": 0.28, "/fspo": 0.22, "/fdpo": 0.22, "/plan": 0.18, "/exams": 0.18},
        "page_type_bonus": {"contacts": 0.42},
        "page_type_penalty": {"directory": 0.28, "hub": 0.12, "plan": 0.16},
        "source_facet_bonus": {"admission_contacts": 0.58, "contacts": 0.2},
        "source_facet_penalty": {},
    },
    "admission_budget": {
        "path_bonus": {"/budget": 0.28, "/plan": 0.34},
        "path_penalty": {"/rules": 0.16},
        "page_type_bonus": {"plan": 0.34, "reference": 0.14},
        "page_type_penalty": {"policy": 0.12},
        "source_facet_bonus": {"admission_budget": 0.42},
        "source_facet_penalty": {},
    },
    "admission_mag": {
        "path_bonus": {"/mag": 0.34},
        "path_penalty": {"/bach": 0.18},
        "page_type_bonus": {"hub": 0.16},
        "page_type_penalty": {"policy": 0.14},
        "source_facet_bonus": {"admission_mag": 0.44},
        "source_facet_penalty": {"admission_bach": 0.18},
    },
    "tuition_price": {
        "path_bonus": {"/price": 0.32, "/paid_edu": 0.22, "/pay_edu": 0.24},
        "path_penalty": {"/monitor": 0.18, "/rekv": 0.18, "/pol_zak": 0.18},
        "page_type_bonus": {"document": 0.38, "detail": 0.16, "reference": 0.12},
        "page_type_penalty": {"hub": 0.08},
        "source_facet_bonus": {"tuition_price": 0.44, "tuition": 0.12},
        "source_facet_penalty": {},
    },
    "tuition_payment": {
        "path_bonus": {"/pay": 0.46, "/pol_usl": 0.42, "/inf_dog": 0.3, "/pay_edu": 0.24},
        "path_penalty": {"/monitor": 0.18, "/price": 0.16, "/form_d": 0.12, "/paid_edu": 0.08},
        "page_type_bonus": {"policy": 0.18, "detail": 0.28, "document": 0.08, "reference": 0.06},
        "page_type_penalty": {"hub": 0.08},
        "source_facet_bonus": {"tuition_payment": 0.5, "tuition": 0.12},
        "source_facet_penalty": {},
    },
    "student_unions": {
        "path_bonus": {"/pposa": 0.42, "/starsovet": 0.42, "/domsovet": 0.32},
        "path_penalty": {"/cyber": 0.18, "/design": 0.18, "/evo": 0.12},
        "page_type_bonus": {"organization": 0.34, "hub": 0.1},
        "page_type_penalty": {"profile": 0.06},
        "source_facet_bonus": {"student_unions": 0.44},
        "source_facet_penalty": {},
    },
    "dorm": {
        "path_bonus": {"/objects": 0.34, "/dom/": 0.28, "/dom/2": 0.42, "/pay_dom": 0.18},
        "path_penalty": {"/inter/": 0.24},
        "page_type_bonus": {"facilities": 0.34, "reference": 0.12},
        "page_type_penalty": {"landing": 0.1},
        "source_facet_bonus": {"dorm": 0.44, "location": 0.1},
        "source_facet_penalty": {},
    },
    "vrmp": {
        "path_bonus": {"/vrmp": 0.34, "/vrmp/tochka": 0.34, "/tochka": 0.22},
        "path_penalty": {"/prioritet2030": 0.26, "/anticor": 0.24, "/eif/": 0.22, "/search": 0.28},
        "page_type_bonus": {"hub": 0.22, "profile": 0.18},
        "page_type_penalty": {"faq": 0.18},
        "source_facet_bonus": {"vrmp": 0.44},
        "source_facet_penalty": {},
    },
    "org_unit": {
        "path_bonus": {"/struct": 0.28, "/department": 0.22, "/dean": 0.22, "/library": 0.28},
        "path_penalty": {"/faq": 0.16, "/news": 0.14, "/pubs": 0.14},
        "page_type_bonus": {"reference": 0.22, "directory": 0.18, "contacts": 0.14},
        "page_type_penalty": {"hub": 0.1, "policy": 0.08},
        "source_facet_bonus": {"org_unit": 0.46},
        "source_facet_penalty": {},
    },
    "org_contacts": {
        "path_bonus": {"/contacts": 0.42, "/contact": 0.38, "/sveden/common": 0.22, "/sveden/managers": 0.18, "/struct": 0.12},
        "path_penalty": {"/faq": 0.16, "/news": 0.12},
        "page_type_bonus": {"contacts": 0.32, "directory": 0.24, "reference": 0.12},
        "page_type_penalty": {"hub": 0.08, "policy": 0.08},
        "source_facet_bonus": {"org_contacts": 0.5, "contacts": 0.2, "org_unit": 0.12},
        "source_facet_penalty": {},
    },
    "org_schedule": {
        "path_bonus": {"/contacts": 0.26, "/contact": 0.22, "/sveden/common": 0.18, "/struct": 0.14},
        "path_penalty": {"/faq": 0.16, "/news": 0.12},
        "page_type_bonus": {"contacts": 0.34, "reference": 0.18, "directory": 0.16},
        "page_type_penalty": {"hub": 0.1, "policy": 0.08},
        "source_facet_bonus": {"org_schedule": 0.58, "org_contacts": 0.16, "org_unit": 0.12},
        "source_facet_penalty": {},
    },
    "org_staff": {
        "path_bonus": {"/managers": 0.44, "/employees": 0.44, "/empbook": 0.34, "/struct": 0.14},
        "path_penalty": {"/faq": 0.14, "/news": 0.12},
        "page_type_bonus": {"directory": 0.38, "reference": 0.12},
        "page_type_penalty": {"hub": 0.08, "policy": 0.08},
        "source_facet_bonus": {"org_staff": 0.54, "org_unit": 0.12},
        "source_facet_penalty": {},
    },
}

INTENT_PROFILES: dict[str, IntentProfile] = {
    "admission": IntentProfile(
        query_hints=(
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
        fact_tokens=frozenset({"баллы", "абитуриент", "документы", "направления", "поступление"}),
        trusted_source_hints=("/abitur", "/priem", "/admission", "/sveden"),
        preferred_source_hints=("priem.guap.ru", "/abitur", "/bach", "/mag", "/dates", "/contacts"),
        query_boost_terms=("приемная комиссия", "документы", "сроки приема"),
        display_summary=(
            "Условия поступления зависят от программы и уровня обучения. "
            "Проверьте актуальные правила и сроки в разделе для абитуриентов."
        ),
        speaker_summary="Проверьте актуальные правила и сроки в разделе для абитуриентов ГУАП.",
        min_evidence_for_freeform=0.32,
        min_fact_token_hits=2,
        priority=90,
    ),
    "tuition": IntentProfile(
        query_hints=(
            "стоим",
            "стои",
            "оплат",
            "цена",
            "обучен",
            "платн",
            "договор",
            "контракт",
        ),
        fact_tokens=frozenset({"стоимость", "цена", "оплата", "договор", "руб", "обучение", "платное"}),
        trusted_source_hints=(
            "/eif/pay",
            "/eif/inf_dog",
            "/eif/price",
            "/eif/pol_usl",
            "/sveden/paid_edu",
            "/priem",
            "/abitur",
        ),
        preferred_source_hints=("/eif/", "/sveden/pay_edu", "/sveden/paid_edu", "/price", "/pay", "/pol_usl"),
        allowed_source_hints=(
            "/eif/pay",
            "/eif/inf_dog",
            "/eif/price",
            "/eif/pol_usl",
            "/sveden/paid_edu",
            "/priem",
            "/abitur",
        ),
        denied_source_hints=("/sveden/inter",),
        noisy_sentence_markers=(
            "кнр",
            "сиань",
            "обучение заграницей",
            "договор о сотрудничестве",
            "мониторинг до",
        ),
        query_boost_terms=("стоимость", "оплата", "договор", "платное образование"),
        display_summary=(
            "**Стоимость обучения в ГУАП зависит от программы и формы обучения.**\n"
            "Актуальные цены и условия оплаты смотрите в разделе платных образовательных услуг."
        ),
        speaker_summary="Стоимость зависит от программы и формы; актуально в разделе платных услуг ГУАП.",
        min_evidence_for_freeform=0.3,
        min_fact_token_hits=2,
        priority=85,
    ),
    "organization": IntentProfile(
        query_hints=(
            "отдел",
            "кадр",
            "декан",
            "деканат",
            "подраздел",
            "управлен",
            "служб",
            "библиот",
            "кафедр",
            "режим",
            "график",
            "часы работы",
            "расписание",
        ),
        fact_tokens=frozenset(
            {
                "отдел",
                "контакты",
                "телефон",
                "почта",
                "адрес",
                "режим",
                "работы",
                "сотрудники",
                "кабинет",
                "деканат",
                "кадры",
                "подразделение",
                "библиотека",
            }
        ),
        trusted_source_hints=(
            "/contacts",
            "/contact",
            "/sveden/common",
            "/sveden/managers",
            "/employees",
            "/managers",
            "/struct",
            "lib.guap.ru",
            "/empbook",
        ),
        preferred_source_hints=(
            "/contacts",
            "/contact",
            "/sveden/common",
            "/sveden/managers",
            "/employees",
            "/managers",
            "/struct",
            "lib.guap.ru",
            "/empbook",
        ),
        query_boost_terms=("контакты подразделения", "режим работы", "часы работы", "сотрудники подразделения"),
        display_summary="Точные контакты, режим работы и сотрудников подразделений лучше смотреть на официальной странице подразделения ГУАП.",
        speaker_summary="Точные контакты и режим работы лучше смотреть на официальной странице подразделения ГУАП.",
        min_evidence_for_freeform=0.34,
        min_fact_token_hits=2,
        priority=80,
    ),
    "contacts": IntentProfile(
        query_hints=("контакт", "связ", "телефон", "почт", "email", "приемн"),
        fact_tokens=frozenset({"контакты", "телефон", "почта", "email", "комиссия", "приемная"}),
        trusted_source_hints=(
            "/sveden/common",
            "/contacts",
            "/contact",
            "/abitur",
            "new.guap.ru/",
            "/empbook",
            "lib.guap.ru",
            "/med/struct",
        ),
        preferred_source_hints=(
            "priem.guap.ru/contacts",
            "/contacts",
            "/contact",
            "/sveden/common",
            "/sveden/managers",
        ),
        query_boost_terms=("контакты", "телефон", "адрес"),
        display_summary="Актуальные контакты лучше смотреть в официальном разделе сведений и контактов ГУАП.",
        speaker_summary="Актуальные контакты смотрите в официальном разделе контактов ГУАП.",
        priority=70,
    ),
    "location": IntentProfile(
        query_hints=("где", "адрес", "корпус", "кампус", "наход"),
        fact_tokens=frozenset({"адрес", "корпус", "кампус"}),
        trusted_source_hints=("/sveden/common", "/contacts", "/address", "/objects"),
        preferred_source_hints=("/sveden/common", "/sveden/objects", "/address", "/contacts"),
        query_boost_terms=("адрес", "корпус", "кампус"),
        display_summary="Адреса и корпуса университета доступны в официальном разделе сведений ГУАП.",
        speaker_summary="Актуальные адреса и корпуса смотрите в официальных сведениях ГУАП.",
        priority=60,
    ),
    "student_life": IntentProfile(
        query_hints=("студен", "актив", "круж", "клуб", "театр", "спорт"),
        fact_tokens=frozenset({"студенты", "клуб", "театр", "спорт", "активности"}),
        trusted_source_hints=("/studlife", "/students", "/clubs", "/sport"),
        preferred_source_hints=("/studlife", "/vrmp"),
        denied_source_hints=("/it/service/",),
        query_boost_terms=("студенческие активности", "клубы", "спорт", "театр"),
        display_summary="Студенческие активности включают клубы, спорт и творческие объединения ГУАП.",
        speaker_summary="Студенческие активности ГУАП включают клубы, спорт и творческие объединения.",
        priority=50,
    ),
    "official_info": IntentProfile(
        query_hints=("сведен", "официальн", "документ", "лиценз", "аккредит"),
        trusted_source_hints=("/sveden", "/document"),
        preferred_source_hints=("/sveden", "/document"),
        priority=40,
    ),
}


def _low_text(title: str, text: str) -> str:
    return f"{title.lower()}\n{text.lower()}"


def _path(canonical_url: str) -> str:
    return urlparse(canonical_url).path.rstrip("/") or "/"


def _host(canonical_url: str) -> str:
    return urlparse(canonical_url).netloc.lower()


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def looks_like_org_unit(canonical_url: str, title: str, text: str) -> bool:
    path = _path(canonical_url).lower()
    low = _low_text(title, text)
    return _has_any(path, ORGANIZATION_PATH_MARKERS) or _has_any(low, ORGANIZATION_MARKERS)


def classify_page_type(source_type: str, canonical_url: str, title: str, text: str = "") -> str:
    path = _path(canonical_url)
    low_title = title.lower()
    low = _low_text(title, text)
    if source_type == "doc":
        return "document"
    if path.endswith("/faq") or "вопросы и ответы" in low_title:
        return "faq"
    if (
        path.endswith("/contacts")
        or "как нас найти" in low_title
        or "приемная комиссия" in low
        or _has_any(low, ORGANIZATION_SCHEDULE_MARKERS)
    ):
        return "contacts"
    if path.endswith("/rules") or "правила приема" in low_title or "положение" in low_title:
        return "policy"
    if path.endswith("/dates") or "сроки проведения приема" in low_title:
        return "schedule"
    if path.endswith("/calc") or "калькулятор" in low_title:
        return "catalog"
    if path.endswith("/plan") or path.endswith("/budget") or "количество мест" in low_title:
        return "plan"
    if path.endswith("/managers") or path.endswith("/employees") or path.startswith("/empbook"):
        return "directory"
    if path.endswith("/common") or path.endswith("/document") or path.endswith("/paid_edu") or path.endswith("/pay_edu"):
        return "reference"
    if path.startswith("/dom/") or path.endswith("/objects") or "общежит" in low_title or "материально-техническое обеспечение" in low_title:
        return "facilities"
    if any(path == value for value in ("/studlife", "/vrmp", "/sveden", "/eif", "/bach", "/mag", "/")):
        return "hub"
    if any(path.endswith(value) for value in ("/pposa", "/starsovet", "/domsovet")):
        return "organization"
    if path.startswith("/studlife/") or path.startswith("/vrmp/"):
        return "profile"
    if looks_like_org_unit(canonical_url, title, text):
        if _has_any(low, ORGANIZATION_STAFF_MARKERS):
            return "directory"
        if _has_any(low, ORGANIZATION_CONTACT_MARKERS) or _has_any(low, ORGANIZATION_SCHEDULE_MARKERS):
            return "contacts"
        return "reference"
    return "detail"


def classify_source_facets(canonical_url: str, title: str, text: str) -> list[str]:
    facets: set[str] = set()
    host = _host(canonical_url)
    path = _path(canonical_url)
    low_title = title.lower()
    low_text = text.lower()
    full_text = f"{low_title}\n{low_text}"

    if host == "priem.guap.ru" or path.startswith("/bach") or path.startswith("/mag"):
        facets.add("admission")
    if path.startswith("/bach"):
        facets.add("admission_bach")
    if path.startswith("/mag"):
        facets.add("admission_mag")
    if path.endswith("/calc") or "калькулятор" in low_title:
        facets.add("admission_directions")
    if path.endswith("/dates") or "сроки" in low_title:
        facets.add("admission_dates")
    if host == "priem.guap.ru" and (
        path.endswith("/contacts") or "приемная комиссия" in full_text or "как нас найти" in full_text
    ):
        facets.add("admission_contacts")
        facets.add("contacts")
    elif (
        path.endswith("/contacts")
        or "контакты" in full_text
        or "телефон:" in full_text
        or "эл. почта" in full_text
        or "@guap.ru" in full_text
    ):
        facets.add("contacts")
    if path.endswith("/budget") or path.endswith("/plan") or "количество мест" in low_title:
        facets.add("admission_budget")
    if path.startswith("/eif") or path.startswith("/sveden/pay") or path.startswith("/sveden/paid"):
        facets.add("tuition")
    if (
        path.endswith("/price")
        or path.endswith("/paid_edu")
        or path.endswith("/pay_edu")
        or "цены обучения" in low_text
        or "стоимость обучения" in low_text
    ):
        facets.add("tuition_price")
    if path.endswith("/pay") or path.endswith("/pol_usl") or path.endswith("/inf_dog") or "оплата обучения" in low_title:
        facets.add("tuition_payment")
    if path.startswith("/sveden"):
        facets.add("official_info")
    if path.endswith("/common") or "адрес" in low_title:
        facets.add("location")
    if path.startswith("/dom/") or path.endswith("/objects") or "общежит" in full_text:
        facets.add("dorm")
        facets.add("location")
    if path.startswith("/studlife"):
        facets.add("student_life")
    if any(path.endswith(value) for value in ("/pposa", "/starsovet", "/domsovet")) or any(
        marker in low_title for marker in ("самоуправление", "профком", "совет старост", "советы общежитий")
    ):
        facets.add("student_unions")
    if path.startswith("/vrmp") or "врмп" in low_title:
        facets.add("vrmp")

    org_unit = looks_like_org_unit(canonical_url, title, text) or host == "lib.guap.ru"
    if org_unit:
        facets.add("org_unit")
    if path.endswith("/managers") or path.endswith("/employees") or path.startswith("/empbook") or _has_any(full_text, ORGANIZATION_STAFF_MARKERS):
        facets.add("org_staff")
        facets.add("org_unit")
    if org_unit and (_has_any(full_text, ORGANIZATION_CONTACT_MARKERS) or path.endswith("/contacts")):
        facets.add("org_contacts")
        facets.add("contacts")
    if org_unit and _has_any(full_text, ORGANIZATION_SCHEDULE_MARKERS):
        facets.add("org_schedule")
    if org_unit and ("адрес" in full_text or "кабинет" in full_text):
        facets.add("location")

    return sorted(facets)


def is_officialish_page(
    *,
    source_type: str,
    canonical_url: str,
    title: str,
    text: str,
    page_type: str,
    source_facets: list[str],
) -> bool:
    if source_type == "doc":
        return True
    host = _host(canonical_url)
    low = _low_text(title, text)
    if not host.endswith("guap.ru"):
        return False
    if source_facets:
        return True
    if page_type in {"contacts", "directory", "reference", "schedule", "policy", "plan", "facilities", "organization", "hub"}:
        return True
    if looks_like_org_unit(canonical_url, title, text):
        return True
    if _has_any(low, ORGANIZATION_CONTACT_MARKERS) or _has_any(low, ORGANIZATION_SCHEDULE_MARKERS):
        return True
    return False
