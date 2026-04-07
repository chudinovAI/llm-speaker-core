from __future__ import annotations

import re

from llm_speaker_core.taxonomy import FACET_HINTS, INTENT_PROFILES, QUERY_EXPANSIONS

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


INTENT_HINTS: dict[str, tuple[str, ...]] = {
    intent: profile.query_hints for intent, profile in INTENT_PROFILES.items()
}


def detect_intents(text: str) -> list[str]:
    low = text.lower()
    scored: list[tuple[int, int, str]] = []
    for intent, profile in INTENT_PROFILES.items():
        matches = sum(1 for hint in profile.query_hints if hint in low)
        if matches > 0:
            scored.append((matches, profile.priority, intent))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [intent for _, _, intent in scored] or ["general"]


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
    if "org_contacts" in facets and "contacts" not in facets:
        facets.append("contacts")
    if "org_schedule" in facets and "org_unit" not in facets:
        facets.append("org_unit")
    if "org_staff" in facets and "org_unit" not in facets:
        facets.append("org_unit")
    if "tuition_price" in facets and "tuition_payment" in facets:
        if any(token.startswith(("скольк", "стоим", "стои", "цена")) for token in expanded_tokens):
            facets = [facet for facet in facets if facet != "tuition_payment"] + ["tuition_price"]
    return sorted(set(facets))
