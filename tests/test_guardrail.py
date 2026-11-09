"""P0.6/P0.9 — guardrail parser, prefilter, and fail-safe behavior."""
import pytest

from src.safety import SafetyLabel, looks_like_emergency, parse_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SAFE", SafetyLabel.SAFE),
        ("safe", SafetyLabel.SAFE),
        ("  SAFE  ", SafetyLabel.SAFE),
        ("UNSAFE", SafetyLabel.UNSAFE),
        ("EMERGENCY", SafetyLabel.EMERGENCY),
        ("NEEDS_CLARIFICATION", SafetyLabel.NEEDS_CLARIFICATION),
        # Adversarial / malformed — must never substring-route:
        ("SAFE \u2014 not unsafe", SafetyLabel.SAFE),  # first token wins
        ("UNSAFE?", SafetyLabel.UNSAFE),  # trailing punctuation tolerated
        ("", SafetyLabel.NEEDS_CLARIFICATION),
        ("   ", SafetyLabel.NEEDS_CLARIFICATION),
        ("garbage output with no label", SafetyLabel.NEEDS_CLARIFICATION),
        ("SAFE and UNSAFE", SafetyLabel.SAFE),  # strict first-token, not `in`
        ("I think this is UNSAFE content", SafetyLabel.NEEDS_CLARIFICATION),
        (None, SafetyLabel.NEEDS_CLARIFICATION),
        (123, SafetyLabel.NEEDS_CLARIFICATION),
    ],
)
def test_parse_label(raw, expected):
    assert parse_label(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I have severe chest pain",
        "I can't breathe, help",
        "there is heavy bleeding from the cut",
        "he is unconscious after the fall",
        "signs of stroke, face drooping",
        "I want to kill myself",
        "I took an overdose of pills",
    ],
)
def test_prefilter_catches_emergencies(text):
    assert looks_like_emergency(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "What are the clinic timings?",
        "Is Dr. Meera available on Sunday?",
        "How long should I fast before a blood sugar test?",
        "What is the consultation fee?",
        "",
    ],
)
def test_prefilter_passes_benign(text):
    assert looks_like_emergency(text) is False


def test_guardrail_exception_fails_safe_not_crash():
    """LLM raising (timeout/404/outage) must route to clarify, not raise."""
    from langchain_core.messages import HumanMessage

    from src.agent import build_medical_agent

    class ExplodingLLM:
        def invoke(self, *a, **k):
            raise TimeoutError("groq timeout")

    graph = build_medical_agent(retriever=_FakeRetriever(), llm=ExplodingLLM())
    # Benign input + dead classifier -> clarify canned message, HTTP-200-safe.
    out = graph.invoke(
        {"messages": [HumanMessage(content="What are the clinic timings?")]},
        config={"configurable": {"thread_id": "t-guardrail-exc"}},
    )
    assert "rephrase" in out["messages"][-1].content.lower()
    # Emergency input + dead classifier -> emergency guidance via prefilter.
    out2 = graph.invoke(
        {"messages": [HumanMessage(content="severe chest pain, help me")]},
        config={"configurable": {"thread_id": "t-guardrail-emg"}},
    )
    assert "emergency" in out2["messages"][-1].content.lower()


def test_guardrail_garbage_label_does_not_crash():
    """Garbage LLM output -> clarify path, never an exception."""
    from langchain_core.messages import HumanMessage

    from src.agent import build_medical_agent

    class GarbageLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, msgs, *a, **k):
            self.calls += 1
            content = "Processing request... please hold"  # no label

            class R:
                pass

            r = R()
            r.content = content
            return r

    graph = build_medical_agent(retriever=_FakeRetriever(), llm=GarbageLLM())
    out = graph.invoke(
        {"messages": [HumanMessage(content="How do I book an appointment?")]},
        config={"configurable": {"thread_id": "t-guardrail-garbage"}},
    )
    assert out["messages"][-1].content  # any safe canned reply, no raise


class _FakeRetriever:
    """New-architecture fake: structured search() (P1.1), no @tool fiction."""

    def search(self, query):
        from src.evidence import Evidence, RetrievalResult, SUCCESS

        return RetrievalResult(
            SUCCESS,
            [Evidence(text="Clinic is open 9am-5pm.", score=5.0,
                      document_name="Clinic Guide", page=0)],
        )


def test_guardrail_404_like_failure_routes_safe():
    """Simulates the user's groq 404 model_not_found during classification."""
    from langchain_core.messages import HumanMessage

    from src.agent import build_medical_agent

    class NotFoundLLM:
        def invoke(self, *a, **k):
            raise RuntimeError("404 model_not_found")

    graph = build_medical_agent(retriever=_FakeRetriever(), llm=NotFoundLLM())
    out = graph.invoke(
        {"messages": [HumanMessage(content="What is the consultation fee?")]},
        config={"configurable": {"thread_id": "t-guardrail-404"}},
    )
    text = out["messages"][-1].content.lower()
    assert "traceback" not in text
    assert ("rephrase" in text) or ("emergency" in text) or ("could not find" in text)
