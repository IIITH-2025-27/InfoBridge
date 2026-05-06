"""Dictionary-based detection and replies for greeting/general queries."""

import re
from typing import Optional


GENERAL_QUERY_DICTIONARY = {
    "greeting": {
        "en": (
            "hi",
            "hii",
            "hello",
            "hey",
            "hello there",
            "hi there",
            "good morning",
            "good afternoon",
            "good evening",
            "good day",
            "greetings",
            "how can you help",
            "how can you assist",
            "how can you help me",
            "how can you assist me",
            "what can you do",
            "what do you do?",
            "how are you"
        ),
        "hi": (
            "हाय",
            "हाई",
            "हेलो",
            "hello",
            "hi",
            "नमस्ते",
            "नमस्कार",
            "सुप्रभात",
            "शुभ दोपहर",
            "शुभ संध्या",
        ),
    },
    "capabilities": {
        "en": (
            "what can you do",
            "what do you do",
            "who are you",
            "tell me about yourself",
            "what services do you cover",
            "what services do you offer",
            "how can you help",
        ),
        "hi": (
            "आप क्या कर सकते हैं",
            "आप क्या करते हैं",
            "आप कौन हैं",
            "अपने बारे में बताओ",
            "आप किन सेवाओं को कवर करते हैं",
            "आप कैसे मदद कर सकते हैं",
        ),
    },
    "thanks": {
        "en": (
            "thanks",
            "thank you",
            "thx",
            "ty",
            "dhanyavaad",
            "dhanyavad",
        ),
        "hi": (
            "धन्यवाद",
            "शुक्रिया",
            "बहुत धन्यवाद",
            "थैंक यू",
            "धन्यावाद",
        ),
    },
    "farewell": {
        "en": (
            "bye",
            "goodbye",
            "good bye",
            "see you",
            "see ya",
            "take care",
        ),
        "hi": (
            "अलविदा",
            "फिर मिलेंगे",
            "बाय",
            "टेक केयर",
        ),
    },
}


GENERAL_QUERY_RESPONSES = {
    "en": {
        "greeting": "Hello! Ask me about Passport, Voter ID, Driving Licence, Income Tax, or Ayushman Bharat.",
        "capabilities": "I can help with Passport, Voter ID, Driving Licence, Income Tax, and Ayushman Bharat questions.",
        "thanks": "You're welcome. Ask me anything about the supported services.",
        "farewell": "Goodbye! Come back if you need help with a government service.",
    },
    "hi": {
        "greeting": "नमस्ते! आप पासपोर्ट, वोटर आईडी, ड्राइविंग लाइसेंस, इनकम टैक्स या आयुष्मान भारत के बारे में पूछ सकते हैं।",
        "capabilities": "मैं पासपोर्ट, वोटर आईडी, ड्राइविंग लाइसेंस, इनकम टैक्स और आयुष्मान भारत से जुड़े प्रश्नों में मदद कर सकता हूँ।",
        "thanks": "आपका स्वागत है। समर्थित सेवाओं के बारे में कुछ भी पूछिए।",
        "farewell": "अलविदा! जब भी सरकारी सेवाओं में मदद चाहिए, वापस आइए।",
    },
}


def _normalize_query(query: str) -> str:
    normalized = (query or "").strip().lower()
    normalized = re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def match_general_query(query: str) -> Optional[str]:
    """Return the matching general-query category, if any."""
    normalized = _normalize_query(query)
    if not normalized:
        return None

    for category, variants_by_language in GENERAL_QUERY_DICTIONARY.items():
        for variants in variants_by_language.values():
            for variant in variants:
                normalized_variant = _normalize_query(variant)
                if not normalized_variant:
                    continue
                if normalized == normalized_variant or normalized.startswith(f"{normalized_variant} "):
                    return category

    return None


def get_general_query_response(query: str, language: str = "en") -> Optional[str]:
    """Return a dictionary-based reply for greetings/general queries."""
    category = match_general_query(query)
    if not category:
        return None

    selected_language = (language or "en").strip().lower()
    if selected_language not in {"en", "hi"}:
        selected_language = "en"

    language_responses = GENERAL_QUERY_RESPONSES.get(selected_language, {})
    return language_responses.get(category)