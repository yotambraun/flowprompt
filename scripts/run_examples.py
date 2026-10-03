"""Run every script in examples/ offline against FlowPrompt's FakeLLM.

Each example runs in a fresh subprocess, in a temporary working directory,
with a placeholder API key so the "live" code paths execute too. LLM calls
are answered locally by :class:`flowprompt.testing.FakeLLM`; nothing leaves
the machine. Pytest-style examples (``*pytest*.py``) are run with pytest.

Usage:
    uv run python scripts/run_examples.py            # all examples
    uv run python scripts/run_examples.py 06 10      # a subset (name filters)

Exits non-zero if any example fails, so it can gate CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

_BOOTSTRAP = r"""
import runpy, sys
from flowprompt.testing import FakeLLM

def responder(messages):
    text = " ".join(str(m.get("content", "")) for m in messages).lower()
    for word in ("negative", "positive", "neutral"):
        if word in text.split("text:")[-1]:
            return word
    return None  # default: echo, or schema-shaped JSON for structured prompts

path = sys.argv[1]
with FakeLLM(responder):
    if "pytest" in path:
        import pytest
        sys.exit(pytest.main([path, "-q", "-p", "no:cacheprovider", "--no-cov"]))
    runpy.run_path(path, run_name="__main__")
"""


def discover(filters: list[str]) -> list[Path]:
    paths = sorted(EXAMPLES.glob("[0-9]*.py"))
    if filters:
        paths = [p for p in paths if any(f in p.name for f in filters)]
    return paths


def run_example(path: Path, timeout: float = 120.0) -> tuple[bool, float, str]:
    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", "sk-placeholder-for-offline-examples")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if p
    )
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, "-c", _BOOTSTRAP, str(path)],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    elapsed = time.perf_counter() - start
    output = proc.stdout + proc.stderr
    return proc.returncode == 0, elapsed, output


def main(argv: list[str]) -> int:
    paths = discover(argv)
    if not paths:
        print("No examples found.")
        return 1
    failures = 0
    for path in paths:
        ok, elapsed, output = run_example(path)
        status = "ok  " if ok else "FAIL"
        print(f"{status} {path.name:<28} {elapsed:5.1f}s")
        if not ok:
            failures += 1
            print("    " + "\n    ".join(output.strip().splitlines()[-25:]))
    print(f"\n{len(paths) - failures}/{len(paths)} examples passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
