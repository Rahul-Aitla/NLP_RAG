"""P0.5/P0.9 — unified safety policy: image path must not diagnose/treat."""
from src.safety import IMAGE_DISCLAIMER, IMAGE_SYSTEM_PROMPT, MEDICAL_SAFETY_POLICY

# Directive-style phrases from the OLD image prompt ("suggest possible
# conditions ... recommend treatments"). Bare words like "prescribe" also
# appear inside prohibitions ("You must NEVER prescribe ..."), which is
# correct — so this list targets request-style directives only.
FORBIDDEN_DIRECTIVES = [
    "suggest possible conditions",
    "recommend treatments",
    "suggest ... medication",
    "provide a diagnosis",
    "give a diagnosis",
    "tell the user what condition",
]


def test_image_prompt_never_requests_diagnosis_or_treatment():
    lowered = IMAGE_SYSTEM_PROMPT.lower()
    for phrase in FORBIDDEN_DIRECTIVES:
        assert phrase not in lowered, f"image prompt must not contain: {phrase}"


def test_image_prompt_contains_prohibitions():
    lowered = IMAGE_SYSTEM_PROMPT.lower()
    assert "never diagnose" in lowered
    assert "do not" in lowered and "medication" in lowered
    assert "healthcare professional" in lowered


def test_shared_policy_forbids_certainty_and_dosing():
    lowered = MEDICAL_SAFETY_POLICY.lower()
    assert "never diagnose" in lowered
    assert "dosage" in lowered
    assert "certainty" in lowered


def test_disclaimer_present_and_nondiagnostic():
    assert "not a diagnosis" in IMAGE_DISCLAIMER
    assert "healthcare professional" in IMAGE_DISCLAIMER
