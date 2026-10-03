"""The examples and the README must keep working exactly as written.

LLM calls are answered offline by FakeLLM, so these tests need no API key.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("[0-9]*.py"))


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_examples", ROOT / "scripts" / "run_examples.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_examples"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_runs_offline(path: Path) -> None:
    runner = _load_runner()
    ok, _elapsed, output = runner.run_example(path)
    assert ok, output[-4000:]


def _python_blocks(markdown: str) -> list[str]:
    """Python code blocks, skipping ones preceded by <!-- not-run -->."""
    blocks = []
    pattern = re.compile(r"(<!-- not-run -->\s*)?```python\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(markdown):
        if match.group(1):
            continue
        blocks.append(match.group(2))
    return blocks


def _sentiment_responder(
    messages: list[dict[str, Any]], request: dict[str, Any]
) -> Any:
    if request.get("response_format"):
        return None  # schema-shaped JSON for structured prompts
    text = str(messages[-1].get("content", "")).lower()
    if any(w in text for w in ("love", "great", "best", "amazing")):
        return "positive"
    if any(w in text for w in ("hate", "awful", "worst", "broken")):
        return "negative"
    return None


@pytest.mark.parametrize("doc", ["README.md"])
def test_markdown_python_blocks_run(
    doc: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flowprompt.testing import FakeLLM

    monkeypatch.chdir(tmp_path)
    blocks = _python_blocks((ROOT / doc).read_text(encoding="utf-8"))
    assert blocks, f"no runnable python blocks found in {doc}"
    namespace: dict[str, Any] = {"__name__": "__readme__"}
    with FakeLLM(_sentiment_responder):
        for i, block in enumerate(blocks):
            try:
                exec(compile(block, f"{doc}[block {i}]", "exec"), namespace)
            except Exception as exc:  # pragma: no cover - message for humans
                pytest.fail(f"{doc} block {i} failed: {exc!r}\n---\n{block}")
