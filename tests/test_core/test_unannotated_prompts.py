"""Prompts written exactly like the README: no type annotations on fields.

This module deliberately does NOT use ``from __future__ import annotations``
so that it exercises the same class-creation path as a user's script.
"""

import pytest
from pydantic import BaseModel

from flowprompt import Field, Prompt


def test_readme_quickstart_prompt_is_definable() -> None:
    class ExtractUser(Prompt):
        system = "Extract user info from text."
        user = "Text: {text}"

        class Output(BaseModel):
            name: str
            age: int

    messages = ExtractUser(text="John is 25").to_messages()
    assert messages == [
        {"role": "system", "content": "Extract user info from text."},
        {"role": "user", "content": "Text: John is 25"},
    ]
    assert ExtractUser(text="x").output_model is ExtractUser.Output


def test_unannotated_fields_are_real_string_fields() -> None:
    class Concise(Prompt):
        system = "Be concise."
        user = "Summarize: {text}"

    assert Concise.model_fields["system"].annotation is str
    assert Concise.model_fields["user"].annotation is str
    assert Concise.model_fields["system"].default == "Be concise."
    assert Concise(text="a").user == "Summarize: a"


def test_unannotated_field_with_flowprompt_field() -> None:
    class Templated(Prompt):
        system = "S"
        user = Field(default="Hello {name}", description="greeting")

    assert Templated(name="Ada").user == "Hello Ada"


def test_annotated_style_still_works() -> None:
    class Annotated(Prompt):
        system: str = "A"
        user: str = "B {x}"

    assert Annotated(x=1).to_messages()[1]["content"] == "B 1"


def test_subclass_of_subclass_inherits_declared_fields() -> None:
    class Base(Prompt):
        system = "base"
        user = "u {t}"
        context: str = ""

    class Child(Base):
        system = "child"
        context = "extra context"

    child = Child(t=1)
    assert child.system == "child"
    assert child.user == "u 1"
    assert child.context == "extra context"


def test_unannotated_non_string_value_still_validated() -> None:
    # Overriding a declared ``str`` field with a non-string must still fail
    # validation rather than silently changing the field type.
    from pydantic import ValidationError

    class Bad(Prompt):
        system = 123  # type: ignore[assignment]

    with pytest.raises(ValidationError):
        Bad()


def test_methods_and_properties_are_not_turned_into_fields() -> None:
    class WithHelpers(Prompt):
        system = "S"
        user = "U"

        def helper(self) -> str:
            return "ok"

    p = WithHelpers()
    assert p.helper() == "ok"
    assert "helper" not in WithHelpers.model_fields
