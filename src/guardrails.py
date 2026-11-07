"""
Guardrails (Phase 5)

1. PII redaction: Aadhaar, PAN, mobile numbers and e-mail addresses typed by the
   user are masked before the text reaches the LLM, the conversation memory or the
   logs. The bot never needs these values to explain a procedure.
2. Prompt-injection detection: messages that try to override the bot's rules
   ("ignore previous instructions", "show your system prompt", ...) are refused.
   Retrieved documents are additionally marked as data-only in the system prompt.
"""

import re
from dataclasses import dataclass, field

# ─── PII ─────────────────────────────────────────────────────────────────────

# Verhoeff tables: every real Aadhaar number passes this checksum, so random
# 12-digit numbers (application IDs, amounts) are not redacted by mistake.
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff_valid(number: str) -> bool:
    check = 0
    for i, digit in enumerate(reversed(number)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(digit)]]
    return check == 0


_AADHAAR = re.compile(r"(?<!\d)([2-9]\d{3})[\s-]?(\d{4})[\s-]?(\d{4})(?!\d)")
# 4th character of a PAN is the holder type (P = person, C = company, ...)
_PAN = re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b", re.IGNORECASE)
_MOBILE = re.compile(r"(?<![\d+])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


@dataclass
class RedactionResult:
    text: str
    found: list[str] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return bool(self.found)


def redact_pii(text: str) -> RedactionResult:
    """Mask Aadhaar / PAN / mobile / e-mail values in `text`."""
    found: list[str] = []

    def _aadhaar(match: re.Match) -> str:
        digits = "".join(match.groups())
        if not _verhoeff_valid(digits):
            return match.group(0)
        found.append("aadhaar")
        return "[AADHAAR REDACTED]"

    def _tag(kind: str, label: str):
        def _replace(_match: re.Match) -> str:
            found.append(kind)
            return label
        return _replace

    text = _AADHAAR.sub(_aadhaar, text or "")
    text = _PAN.sub(_tag("pan", "[PAN REDACTED]"), text)
    text = _MOBILE.sub(_tag("mobile", "[MOBILE REDACTED]"), text)
    text = _EMAIL.sub(_tag("email", "[EMAIL REDACTED]"), text)

    return RedactionResult(text=text, found=sorted(set(found)))


def pii_notice(found: list[str], language: str) -> str:
    names_en = {"aadhaar": "Aadhaar number", "pan": "PAN", "mobile": "mobile number", "email": "e-mail address"}
    names_hi = {"aadhaar": "आधार नंबर", "pan": "पैन", "mobile": "मोबाइल नंबर", "email": "ई-मेल पता"}
    if language == "hi":
        items = ", ".join(names_hi[k] for k in found)
        return f"🔒 आपकी गोपनीयता के लिए आपका {items} हटा दिया गया है। कृपया चैट में निजी जानकारी साझा न करें।"
    items = ", ".join(names_en[k] for k in found)
    return f"🔒 For your privacy, your {items} was removed before processing. Please avoid sharing personal details in chat."


# ─── Prompt injection ────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(previous|above|prior|earlier|all|your|the)\b.{0,20}\b(instructions?|rules?|prompts?|guidelines?)",
    r"\bdisregard\b.{0,40}\b(instructions?|rules?|prompts?|guidelines?)",
    r"\bforget\b.{0,30}\b(instructions?|rules?|everything)",
    r"\b(reveal|show|print|repeat|tell me|what is|display)\b.{0,30}\b(system|hidden|initial|original)\s+(prompt|instructions?|message)",
    r"\byou are (now|no longer)\b",
    r"\bact as\b.{0,30}\b(unrestricted|jailbroken|dan|developer mode|without (rules|restrictions))",
    r"\b(jailbreak|developer mode|do anything now)\b",
    r"\bpretend\b.{0,30}\b(no|without)\b.{0,20}\b(rules|restrictions|limits)",
    r"\bnew instructions?\s*:",
    r"\boverride\b.{0,30}\b(instructions?|rules?|safety)",
    # Hindi / Hinglish
    r"(पिछले|पहले के|सभी)\s*(निर्देश|नियम).{0,20}(भूल|अनदेखा|नज़रअंदाज़|नजरअंदाज)",
    r"सिस्टम\s*प्रॉम्प्ट",
    r"\b(pichle|sare|saare)\s+(instructions?|niyam|nirdesh)\s+(bhool|ignore)",
]
_INJECTION_REGEX = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_REGEX.search(text or ""))


def injection_refusal(language: str) -> str:
    if language == "hi":
        return (
            "मैं केवल भारतीय सरकारी सेवाओं (पासपोर्ट, वोटर आईडी, ड्राइविंग लाइसेंस, आयकर, "
            "आयुष्मान भारत) से जुड़े प्रश्नों में मदद कर सकता हूँ। कृपया अपना प्रश्न पूछें।"
        )
    return (
        "I can only help with questions about Indian government services (Passport, Voter ID, "
        "Driving Licence, Income Tax, Ayushman Bharat). Please ask your question about one of these."
    )
