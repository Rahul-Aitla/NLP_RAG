"""P0.5/P0.6 — single source of truth for healthcare safety policy and guardrail schema.

Both the text RAG path (src/agent.py) and the image path (app.py) import from
here so the refusal/diagnosis policy cannot drift between endpoints.
"""
from __future__ import annotations

import re
from enum import Enum


class SafetyLabel(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    EMERGENCY = "EMERGENCY"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class SafetyCategory(str, Enum):
    GENERAL = "GENERAL"
    CLINIC_INFORMATION = "CLINIC_INFORMATION"
    EMERGENCY = "EMERGENCY"
    MEDICATION = "MEDICATION"
    DOSAGE = "DOSAGE"
    DIAGNOSIS = "DIAGNOSIS"
    UNKNOWN = "UNKNOWN"


# --- Shared medical safety policy (P0.5). ------------------------------------
# The assistant must NOT diagnose, prescribe, dose, or claim certainty.
# It MAY describe cautiously, give general info, urge professional evaluation,
# and escalate emergencies.
MEDICAL_SAFETY_POLICY = (
    "You are a clinic information assistant, not a doctor. "
    "You must NEVER diagnose a condition, prescribe or recommend any treatment "
    "or medication, recommend or interpret any dosage, or claim certainty about "
    "any medical condition. You MAY describe visible characteristics cautiously "
    "in general terms, provide general health information, recommend seeking "
    "prompt evaluation by a qualified healthcare professional, and provide "
    "emergency guidance when there are signs of an urgent situation."
)

# Image path system prompt (P0.5): describe-only, aligned with the text path
# refusal policy. Replaces the old prompt that asked for "possible conditions"
# and "treatments".
IMAGE_SYSTEM_PROMPT = (
    "You are a clinic information assistant, not a doctor. "
    f"{MEDICAL_SAFETY_POLICY} "
    "For the uploaded image: describe only what is generally visible, in "
    "cautious non-diagnostic language. Do NOT name a diagnosis, do NOT suggest "
    "any medication or treatment, do NOT mention any dosage. End every reply "
    "by recommending evaluation by a qualified healthcare professional. "
    "If the image or message suggests an urgent or emergency situation "
    "(severe injury, heavy bleeding, breathing difficulty, signs of stroke, "
    "chest pain, self-harm), tell the user to seek emergency care immediately."
)

IMAGE_DISCLAIMER = (
    " This is general information only, not a diagnosis. "
    "Please consult a qualified healthcare professional for proper evaluation."
)

CLARIFY_MSG = (
    "I'm not sure I understood your question. Could you rephrase it? "
    "I can help with clinic information such as appointments, timings, fees, "
    "doctors, lab tests, reports, and clinic policies."
)

# --- Deterministic emergency pre-filter (P0.6). -------------------------------
# Conservative keyword layer that runs BEFORE the LLM classifier. It never
# diagnoses — it only detects strong emergency indicators so a failed/slow
# classifier can still route to emergency guidance.
_EMERGENCY_PATTERNS = [
    r"\bchest\s*pain\b", r"\bcan'?t\s*breathe\b", r"\btrouble\s*breathing\b",
    r"\bdifficulty\s*breathing\b", r"\bshort(ness)?\s*of\s*breath\b",
    r"\bsevere\s*bleeding\b", r"\bheavy\s*bleeding\b", r"\bblood\s*(won'?t|will\s*not)\s*stop\b",
    r"\bunconscious\b", r"\bpassed\s*out\b", r"\bfainted\b(?!.*test)",
    r"\bstroke\b", r"\bface\s*droop", r"\bslurred\s*speech\b",
    r"\bheart\s*attack\b", r"\bseizure\b", r"\bchok\w*\b",
    r"\bsuicid\w*\b", r"\bkill\s*myself\b", r"\bself[\s-]?harm\b",
    r"\bend\s*my\s*life\b", r"\boverdose\b", r"\bpoison\w*\b",
    r"\bsevere\s*burn", r"\bstabbed\b", r"\bgunshot\b",
    r"\b112\b", r"\b108\b.*\b(emergency|ambulance)\b",
]
_EMERGENCY_RE = re.compile("|".join(f"(?:{p})" for p in _EMERGENCY_PATTERNS), re.IGNORECASE)


def looks_like_emergency(text: str) -> bool:
    """Deterministic, conservative emergency indicator check."""
    if not text:
        return False
    return bool(_EMERGENCY_RE.search(text))


# --- Strict label parser (P0.6). ----------------------------------------------
# Accepts exactly one of the four labels (case-insensitive, surrounding
# whitespace/punctuation tolerated). Anything else -> NEEDS_CLARIFICATION.
# Never substring-match: "SAFE — not unsafe" must NOT route to UNSAFE.
_LABEL_ALIASES = {
    "SAFE": SafetyLabel.SAFE,
    "UNSAFE": SafetyLabel.UNSAFE,
    "EMERGENCY": SafetyLabel.EMERGENCY,
    "NEEDS_CLARIFICATION": SafetyLabel.NEEDS_CLARIFICATION,
    "CLARIFY": SafetyLabel.NEEDS_CLARIFICATION,
    "UNKNOWN": SafetyLabel.NEEDS_CLARIFICATION,
}


def parse_label(raw: object) -> SafetyLabel:
    """Strictly parse an LLM classifier reply into a SafetyLabel."""
    if not isinstance(raw, str):
        return SafetyLabel.NEEDS_CLARIFICATION
    token = raw.strip().upper()
    if not token:
        return SafetyLabel.NEEDS_CLARIFICATION
    # Take the first token and strip common surrounding punctuation.
    token = re.split(r"[\s,;:.!?()\[\]{}\"]+", token, maxsplit=1)[0].strip("'\"-–—")
    return _LABEL_ALIASES.get(token, SafetyLabel.NEEDS_CLARIFICATION)


# Guardrail classifier instruction (4-label schema).
GUARDRAIL_SYSTEM = (
    "You are a safety classifier for a medical clinic assistant. "
    "Classify the user's message into EXACTLY one label:\n"
    "EMERGENCY = a medical emergency (chest pain, severe bleeding, trouble "
    "breathing, unconsciousness, stroke signs, or self-harm/suicide intent).\n"
    "UNSAFE = asks for a diagnosis, a prescription, a drug dosage to take, "
    "or to start/stop/change medication.\n"
    "SAFE = a normal informational question about the clinic (appointments, "
    "timings, fees, doctors, lab tests, policies, reports).\n"
    "NEEDS_CLARIFICATION = the message is unclear, empty, or does not fit "
    "any other label.\n"
    "Reply with ONLY one word: EMERGENCY, UNSAFE, SAFE, or NEEDS_CLARIFICATION."
)
