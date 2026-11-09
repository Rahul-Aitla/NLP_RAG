"""P1.9/P1.20 — GROQ_MODEL is configured, logged, and validated."""
import dataclasses

from src.config import load_settings, require_provider_keys


def test_model_passthrough_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    assert load_settings().groq_model == "openai/gpt-oss-120b"


def test_empty_model_fails_validation():
    s = dataclasses.replace(
        load_settings(), groq_model="  ",
        groq_api_key="x", pinecone_api_key="y",
    )
    try:
        require_provider_keys(s)
    except Exception as exc:  # noqa: BLE001 - asserting message content
        assert "GROQ_MODEL" in str(exc)
    else:
        raise AssertionError("empty GROQ_MODEL must fail validation")


def test_env_example_and_readme_agree(tmp_path=None):
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = open(os.path.join(root, ".env.example")).read()
    readme = open(os.path.join(root, "README.md")).read()
    assert "GROQ_MODEL" in example and "GROQ_MODEL" in readme
