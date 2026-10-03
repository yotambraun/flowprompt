"""YAML/JSON prompt loading for FlowPrompt.

Enables file-based prompt management for:
- Non-developer collaboration
- Version control friendly prompts
- Environment-specific configurations
- Prompt registries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, create_model
from pydantic import Field as PydanticField

from flowprompt.core.prompt import Prompt


class PromptConfig(BaseModel):
    """Configuration for a loaded prompt.

    Attributes:
        name: Unique identifier for the prompt.
        version: Semantic version string.
        description: Human-readable description.
        system: System message template.
        user: User message template.
        output_schema: Optional JSON schema for structured output.
        metadata: Additional metadata (tags, author, etc.).
    """

    name: str
    version: str = "0.0.0"
    description: str = ""
    system: str = "You are a helpful assistant."
    user: str = ""
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = PydanticField(default_factory=dict)

    @classmethod
    def from_yaml(cls, content: str) -> PromptConfig:
        """Load from YAML string."""
        data = yaml.safe_load(content)
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, content: str) -> PromptConfig:
        """Load from JSON string."""
        data = json.loads(content)
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str | Path) -> PromptConfig:
        """Load from a file (YAML or JSON based on extension)."""
        path = Path(path)
        content = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(content)
        elif path.suffix == ".json":
            return cls.from_json(content)
        else:
            # Try YAML first, then JSON
            try:
                return cls.from_yaml(content)
            except yaml.YAMLError:
                return cls.from_json(content)

    def to_yaml(self) -> str:
        """Export to YAML string."""
        return yaml.dump(self.model_dump(exclude_none=True), default_flow_style=False)

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.model_dump(exclude_none=True), indent=indent)

    def to_prompt_class(self) -> type[Prompt[Any]]:
        """Create a Prompt class from this configuration.

        Returns:
            A dynamically created Prompt subclass.
        """
        # Build class attributes
        class_attrs: dict[str, Any] = {
            "__module__": __name__,
            "__qualname__": self.name,
            "__version__": self.version,
            "__doc__": self.description,
            "system": self.system,
            "user": self.user,
        }

        # Add Output class if schema is provided
        if self.output_schema:
            # Create a Pydantic model from JSON schema
            output_class = self._schema_to_model(self.output_schema)
            # Present it as a nested class (``<Name>.Output``) so pydantic does
            # not mistake it for an unannotated field.
            output_class.__module__ = __name__
            output_class.__qualname__ = f"{self.name}.Output"
            class_attrs["Output"] = output_class

        # Create the class dynamically
        prompt_class = type(self.name, (Prompt,), class_attrs)
        return cast(type[Prompt[Any]], prompt_class)

    def _schema_to_model(
        self, schema: dict[str, Any], name: str = "Output"
    ) -> type[BaseModel]:
        """Convert a JSON schema (object) to a Pydantic model.

        Supports the common subset used for structured LLM output: scalar
        types, ``enum``, nested objects, ``array`` with ``items``, nullable
        types, ``required`` and ``description``.
        """
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        field_definitions: dict[str, Any] = {}
        for field_name, prop in properties.items():
            field_type = self._json_type_to_python(prop, f"{name}_{field_name}")
            description = prop.get("description") or None
            if field_name in required:
                field_definitions[field_name] = (
                    field_type,
                    PydanticField(..., description=description),
                )
            else:
                field_definitions[field_name] = (
                    field_type | None,
                    PydanticField(prop.get("default"), description=description),
                )

        model = create_model(name, **field_definitions)
        if schema.get("description"):
            model.__doc__ = schema["description"]
        return model

    def _json_type_to_python(
        self, prop: dict[str, Any] | str, name: str = "Nested"
    ) -> Any:
        """Convert a JSON schema property to a Python type annotation."""
        if isinstance(prop, str):
            prop = {"type": prop}
        if "enum" in prop and prop["enum"]:
            return Literal[tuple(prop["enum"])]
        json_type = prop.get("type", "string")
        if isinstance(json_type, list):
            options = [t for t in json_type if t != "null"]
            inner = self._json_type_to_python(
                {**prop, "type": options[0] if options else "string"}, name
            )
            return inner | None if "null" in json_type else inner
        if json_type == "object" and prop.get("properties"):
            return self._schema_to_model(prop, name.title().replace("_", ""))
        if json_type == "array":
            items = prop.get("items")
            if isinstance(items, dict):
                return list[self._json_type_to_python(items, name + "_item")]  # type: ignore[misc]
            return list[Any]
        type_map: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict[str, Any],
            "null": type(None),
        }
        return type_map.get(json_type, str)


class PromptRegistry:
    """Registry for managing loaded prompts.

    Provides centralized management of prompts loaded from files,
    with support for versioning and hot-reloading.

    Example:
        >>> registry = PromptRegistry("./prompts")
        >>> registry.load_all()
        >>> prompt_class = registry.get("ExtractUser")
        >>> result = prompt_class(text="John is 25").run(model="gpt-4o")
    """

    def __init__(self, prompts_dir: str | Path | None = None) -> None:
        """Initialize the registry.

        Args:
            prompts_dir: Directory containing prompt files.
        """
        self._prompts_dir = Path(prompts_dir) if prompts_dir else None
        self._prompts: dict[str, PromptConfig] = {}
        self._classes: dict[str, type[Prompt[Any]]] = {}

    def register(self, config: PromptConfig) -> type[Prompt[Any]]:
        """Register a prompt configuration.

        Args:
            config: Prompt configuration to register.

        Returns:
            The generated Prompt class.
        """
        self._prompts[config.name] = config
        prompt_class = config.to_prompt_class()
        self._classes[config.name] = prompt_class
        return prompt_class

    def load_file(self, path: str | Path) -> type[Prompt[Any]]:
        """Load and register a prompt from a file.

        Args:
            path: Path to the prompt file.

        Returns:
            The generated Prompt class.
        """
        config = PromptConfig.from_file(path)
        return self.register(config)

    def load_all(self) -> dict[str, type[Prompt[Any]]]:
        """Load all prompts from the prompts directory.

        Returns:
            Dictionary of prompt name to class mappings.
        """
        if self._prompts_dir is None:
            return {}

        loaded: dict[str, type[Prompt[Any]]] = {}
        for ext in ("*.yaml", "*.yml", "*.json"):
            for path in self._prompts_dir.glob(ext):
                try:
                    prompt_class = self.load_file(path)
                    loaded[path.stem] = prompt_class
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")

        return loaded

    def get(self, name: str) -> type[Prompt[Any]] | None:
        """Get a registered prompt class by name.

        Args:
            name: Name of the prompt.

        Returns:
            The Prompt class, or None if not found.
        """
        return self._classes.get(name)

    def get_config(self, name: str) -> PromptConfig | None:
        """Get a prompt configuration by name.

        Args:
            name: Name of the prompt.

        Returns:
            The PromptConfig, or None if not found.
        """
        return self._prompts.get(name)

    def list_prompts(self) -> list[str]:
        """List all registered prompt names."""
        return list(self._prompts.keys())

    def reload(self, name: str) -> type[Prompt[Any]] | None:
        """Reload a prompt from its source file.

        Args:
            name: Name of the prompt to reload.

        Returns:
            The reloaded Prompt class, or None if not found.
        """
        if self._prompts_dir is None:
            return None

        for ext in (".yaml", ".yml", ".json"):
            path = self._prompts_dir / f"{name}{ext}"
            if path.exists():
                return self.load_file(path)

        return None


def load_prompt(path: str | Path) -> type[Prompt[Any]]:
    """Load a single prompt from a file.

    Convenience function for loading individual prompt files.

    Args:
        path: Path to the prompt file.

    Returns:
        A Prompt class configured from the file.

    Example:
        >>> ExtractUser = load_prompt("prompts/extract_user.yaml")
        >>> result = ExtractUser(text="John is 25").run(model="gpt-4o")
    """
    config = PromptConfig.from_file(path)
    return config.to_prompt_class()


def load_prompts(directory: str | Path) -> dict[str, type[Prompt[Any]]]:
    """Load all prompts from a directory.

    Args:
        directory: Directory containing prompt files.

    Returns:
        Dictionary mapping prompt names to classes.

    Example:
        >>> prompts = load_prompts("./prompts")
        >>> result = prompts["ExtractUser"](text="John is 25").run()
    """
    registry = PromptRegistry(directory)
    return registry.load_all()
