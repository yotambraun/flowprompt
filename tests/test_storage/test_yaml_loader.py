"""Tests for loading prompts from YAML / JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from flowprompt import PromptConfig, PromptRegistry, load_prompt, load_prompts

YAML = """
name: SummarizeArticle
version: "1.2.0"
description: Summarize an article
system: You are a precise summarizer.
user: "Summarize: {{ article }}"
output_schema:
  type: object
  properties:
    summary:
      type: string
      description: One-sentence summary
    key_points:
      type: array
      items:
        type: string
    sentiment:
      type: string
      enum: [positive, negative, neutral]
    confidence:
      type: number
    meta:
      type: object
      properties:
        words:
          type: integer
      required: [words]
  required:
    - summary
    - key_points
"""


def test_yaml_prompt_with_output_schema_builds_valid_model(tmp_path: Path) -> None:
    path = tmp_path / "summarize.yaml"
    path.write_text(YAML)
    cls = load_prompt(path)

    prompt = cls(article="Fish glow.")
    assert prompt.to_messages()[1]["content"] == "Summarize: Fish glow."
    assert cls.__version__ == "1.2.0"

    output = cls.Output
    model = output.model_validate(
        {
            "summary": "A fish glows.",
            "key_points": ["deep sea", "light"],
            "sentiment": "neutral",
            "meta": {"words": 3},
        }
    )
    assert model.summary == "A fish glows."
    assert model.key_points == ["deep sea", "light"]
    assert model.confidence is None
    assert model.meta.words == 3
    assert prompt.output_model is output
    # The JSON schema (sent to the model for structured output) is valid.
    schema = output.model_json_schema()
    assert set(schema["required"]) == {"summary", "key_points"}


def test_yaml_schema_is_enforced() -> None:
    cls = PromptConfig.from_yaml(YAML).to_prompt_class()
    with pytest.raises(ValidationError):
        cls.Output.model_validate({"summary": "x"})  # key_points missing
    with pytest.raises(ValidationError):
        cls.Output.model_validate(
            {"summary": "x", "key_points": [], "sentiment": "angry"}
        )


def test_yaml_prompt_without_schema(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("name: Hello\nuser: 'Hi {name}'\n")
    (tmp_path / "b.json").write_text(json.dumps({"name": "Bye", "user": "Bye {name}"}))
    prompts = load_prompts(tmp_path)  # keyed by file name
    assert set(prompts) == {"a", "b"}
    assert prompts["a"](name="Ada").user == "Hi Ada"
    assert prompts["a"](name="Ada").output_model is None
    assert prompts["b"].__name__ == "Bye"


def test_registry_round_trip(tmp_path: Path) -> None:
    config = PromptConfig.from_yaml(YAML)
    (tmp_path / "s.yaml").write_text(config.to_yaml())
    registry = PromptRegistry(tmp_path)
    registry.load_all()
    cls = registry.get("SummarizeArticle")
    assert cls is not None
    assert cls(article="x").system == "You are a precise summarizer."
