"""FlowPrompt CLI - Command-line tools for prompt management."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import typer
    from typer import Argument, Option

    HAS_TYPER = True
except ImportError:
    typer = None  # type: ignore
    Argument = None  # type: ignore
    Option = None  # type: ignore
    HAS_TYPER = False

from flowprompt import __version__


def create_app() -> Any:
    """Create the CLI application."""
    if typer is None:
        raise ImportError(
            "Typer is required for the CLI. "
            "Install it with: pip install flowprompt[dev] or pip install typer"
        )

    app = typer.Typer(
        name="flowprompt",
        help="FlowPrompt - Type-safe prompt management for LLMs",
        add_completion=False,
    )

    @app.command()
    def version() -> None:
        """Show FlowPrompt version."""
        typer.echo(f"FlowPrompt v{__version__}")

    @app.command()
    def init(
        directory: str = Argument(".", help="Directory to initialize"),
        _template: str = Option("basic", help="Project template (basic, advanced)"),
    ) -> None:
        """Initialize a new FlowPrompt project."""
        project_dir = Path(directory)
        prompts_dir = project_dir / "prompts"

        # Create directory structure
        prompts_dir.mkdir(parents=True, exist_ok=True)

        # Create example prompt file
        example_prompt = {
            "name": "ExtractUser",
            "version": "1.0.0",
            "description": "Extract user information from text",
            "system": "You are a precise data extractor. Extract information accurately.",
            "user": "Extract the user information from this text: {{ text }}",
            "output_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The person's name"},
                    "age": {"type": "integer", "description": "The person's age"},
                },
                "required": ["name", "age"],
            },
        }

        example_path = prompts_dir / "extract_user.yaml"
        if not example_path.exists():
            import yaml

            example_path.write_text(yaml.dump(example_prompt, default_flow_style=False))

        # Create config file
        config = {
            "prompts_dir": "prompts",
            "default_model": "gpt-4o",
            "cache": {"enabled": True, "ttl": 3600},
            "tracing": {"enabled": True, "service_name": "flowprompt"},
        }

        config_path = project_dir / "flowprompt.yaml"
        if not config_path.exists():
            import yaml

            config_path.write_text(yaml.dump(config, default_flow_style=False))

        typer.echo(f"Initialized FlowPrompt project in {project_dir.absolute()}")
        typer.echo(f"  Created: {prompts_dir}/")
        typer.echo(f"  Created: {example_path}")
        typer.echo(f"  Created: {config_path}")
        typer.echo("\nNext steps:")
        typer.echo("  1. Edit prompts in the prompts/ directory")
        typer.echo("  2. Run 'flowprompt test' to validate prompts")
        typer.echo("  3. Use prompts in your code with load_prompts()")

    @app.command()
    def test(
        prompts_dir: str = Option("prompts", help="Directory containing prompts"),
        pattern: str = Option("*.yaml", help="Glob pattern for prompt files"),
    ) -> None:
        """Validate and test prompts."""
        from flowprompt.storage.yaml_loader import PromptConfig

        prompts_path = Path(prompts_dir)
        if not prompts_path.exists():
            typer.echo(f"Error: Directory '{prompts_dir}' not found", err=True)
            raise typer.Exit(1)

        passed = 0
        failed = 0

        for path in prompts_path.glob(pattern):
            try:
                config = PromptConfig.from_file(path)
                prompt_class = config.to_prompt_class()

                # Basic validation
                instance = prompt_class()
                instance.to_messages()  # Validates the prompt

                typer.echo(f"  PASS  {path.name} ({config.name} v{config.version})")
                passed += 1
            except Exception as e:
                typer.echo(f"  FAIL  {path.name}: {e}", err=True)
                failed += 1

        typer.echo(f"\n{passed} passed, {failed} failed")
        if failed > 0:
            raise typer.Exit(1)

    @app.command()
    def list_prompts(
        prompts_dir: str = Option(
            "prompts", "--dir", help="Directory containing prompts"
        ),
    ) -> None:
        """List all available prompts."""
        from flowprompt.storage.yaml_loader import PromptConfig

        prompts_path = Path(prompts_dir)
        if not prompts_path.exists():
            typer.echo(f"Error: Directory '{prompts_dir}' not found", err=True)
            raise typer.Exit(1)

        typer.echo("Available prompts:")
        typer.echo("-" * 60)

        for path in sorted(prompts_path.glob("*.yaml")) + sorted(
            prompts_path.glob("*.yml")
        ):
            try:
                config = PromptConfig.from_file(path)
                typer.echo(f"  {config.name:<20} v{config.version:<10} {path.name}")
            except Exception as e:
                typer.echo(f"  Error loading {path.name}: {e}", err=True)

    @app.command()
    def run(
        prompt_file: str = Argument(..., help="Path to prompt file"),
        model: str = Option("gpt-4o", help="Model to use"),
        var: list[str] | None = Option(None, help="Variables (key=value)"),
    ) -> None:
        """Run a prompt from a file."""
        from flowprompt.storage.yaml_loader import load_prompt

        path = Path(prompt_file)
        if not path.exists():
            typer.echo(f"Error: File '{prompt_file}' not found", err=True)
            raise typer.Exit(1)

        # Parse variables
        variables: dict[str, str] = {}
        if var:
            for v in var:
                if "=" in v:
                    key, value = v.split("=", 1)
                    variables[key] = value

        try:
            prompt_class = load_prompt(path)
            instance = prompt_class(**variables)

            typer.echo(f"Running prompt with {model}...")
            result = instance.run(model=model)

            if hasattr(result, "model_dump"):
                typer.echo(json.dumps(result.model_dump(), indent=2))
            else:
                typer.echo(result)

        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    @app.command()
    def stats() -> None:
        """Show tracing statistics."""
        from flowprompt.tracing.otel import get_tracer

        tracer = get_tracer()
        summary = tracer.get_summary()

        typer.echo("FlowPrompt Statistics")
        typer.echo("-" * 40)
        typer.echo(f"Total requests:  {summary['total_requests']}")
        typer.echo(f"Total tokens:    {summary['total_tokens']}")
        typer.echo(f"Total cost:      ${summary['total_cost_usd']:.4f}")
        typer.echo(f"Avg latency:     {summary['avg_latency_ms']:.2f}ms")
        typer.echo(f"Error rate:      {summary['error_rate']:.1%}")

        if summary.get("by_model"):
            typer.echo("\nBy Model:")
            for model, data in summary["by_model"].items():
                typer.echo(
                    f"  {model}: {data['requests']} requests, "
                    f"{data['tokens']} tokens, ${data['cost_usd']:.4f}"
                )

    @app.command()
    def cache_stats() -> None:
        """Show cache statistics."""
        from flowprompt.core.cache import get_cache

        cache = get_cache()
        stats = cache.stats

        typer.echo("Cache Statistics")
        typer.echo("-" * 40)
        typer.echo(f"Enabled:   {stats['enabled']}")
        typer.echo(f"Size:      {stats['size']} entries")
        typer.echo(f"Hits:      {stats['hits']}")
        typer.echo(f"Misses:    {stats['misses']}")
        typer.echo(f"Hit rate:  {stats['hit_rate']:.1%}")

    @app.command()
    def cache_clear() -> None:
        """Clear the prompt cache."""
        from flowprompt.core.cache import get_cache

        cache = get_cache()
        count = cache.clear()
        typer.echo(f"Cleared {count} cache entries")

    @app.command()
    def optimize(
        prompt_file: str = Argument(..., help="Python file containing Prompt class"),
        examples_file: str = Argument(..., help="JSON file with training examples"),
        model: str = Option("gpt-4o-mini", help="Model to use for optimization"),
        strategy: str = Option(
            "fewshot",
            help="Optimization strategy (fewshot, instruction, bootstrap)",
        ),
        output: str = Option(
            None, "--output", "-o", help="Output file for optimized prompt"
        ),
        iterations: int = Option(10, help="Number of optimization iterations"),
    ) -> None:
        """Optimize a prompt using training examples.

        Example:
            flowprompt optimize my_prompt.py examples.json --strategy fewshot

        The examples.json should have this format:
            [
                {"input": {"text": "John is 25"}, "output": {"name": "John", "age": 25}},
                {"input": {"text": "Alice is 30"}, "output": {"name": "Alice", "age": 30}}
            ]
        """
        import importlib.util
        import inspect

        from flowprompt.core.prompt import Prompt
        from flowprompt.optimize import (
            ExactMatch,
            Example,
            ExampleDataset,
            OptimizationConfig,
        )
        from flowprompt.optimize import optimize as run_optimize

        prompt_path = Path(prompt_file)
        examples_path = Path(examples_file)

        if not prompt_path.exists():
            typer.echo(f"Error: Prompt file '{prompt_file}' not found", err=True)
            raise typer.Exit(1)

        if not examples_path.exists():
            typer.echo(f"Error: Examples file '{examples_file}' not found", err=True)
            raise typer.Exit(1)

        # Load the prompt class from Python file
        typer.echo(f"Loading prompt from {prompt_path.name}...")
        try:
            spec = importlib.util.spec_from_file_location("prompt_module", prompt_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load {prompt_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find Prompt subclass in the module
            prompt_class = None
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Prompt) and obj is not Prompt:
                    prompt_class = obj
                    break

            if prompt_class is None:
                typer.echo("Error: No Prompt subclass found in file", err=True)
                raise typer.Exit(1)

            typer.echo(f"  Found: {prompt_class.__name__}")

        except Exception as e:
            typer.echo(f"Error loading prompt: {e}", err=True)
            raise typer.Exit(1) from None

        # Load examples from JSON
        typer.echo(f"Loading examples from {examples_path.name}...")
        try:
            examples_data = json.loads(examples_path.read_text())
            examples = [
                Example(input=ex["input"], output=ex["output"]) for ex in examples_data
            ]
            dataset = ExampleDataset(examples)
            typer.echo(f"  Loaded {len(examples)} examples")
        except Exception as e:
            typer.echo(f"Error loading examples: {e}", err=True)
            raise typer.Exit(1) from None

        # Run baseline evaluation
        typer.echo("\nEvaluating baseline...")
        try:
            from flowprompt.optimize.optimizer import _evaluate_dataset

            baseline_score = _evaluate_dataset(
                prompt_class, dataset, ExactMatch(), model
            )
            typer.echo(f"  Baseline accuracy: {baseline_score:.1%}")
        except Exception as e:
            typer.echo(f"  Could not evaluate baseline: {e}")
            baseline_score = None

        # Run optimization
        typer.echo(f"\nOptimizing with strategy='{strategy}'...")
        typer.echo(f"  Model: {model}")
        typer.echo(f"  Iterations: {iterations}")
        typer.echo("")

        try:
            config = OptimizationConfig(
                max_iterations=iterations,
                num_candidates=3,
            )

            result = run_optimize(
                prompt_class,
                dataset=dataset,
                metric=ExactMatch(),
                model=model,
                strategy=strategy,
                config=config,
            )

            # Show results
            typer.echo("-" * 50)
            typer.echo("OPTIMIZATION COMPLETE")
            typer.echo("-" * 50)
            if baseline_score is not None:
                improvement = result.best_score - baseline_score
                typer.echo(f"  Before: {baseline_score:.1%} accuracy")
                typer.echo(f"  After:  {result.best_score:.1%} accuracy")
                typer.echo(
                    f"  Change: {'+' if improvement >= 0 else ''}{improvement:.1%}"
                )
            else:
                typer.echo(f"  Final accuracy: {result.best_score:.1%}")

            typer.echo(f"  Iterations: {result.iterations}")

            # Save optimized prompt if output specified
            if output:
                output_path = Path(output)
                # Generate optimized prompt code
                optimized_class = result.best_prompt_class

                code = f'''"""Optimized prompt generated by FlowPrompt.

Original: {prompt_path.name}
Strategy: {strategy}
Accuracy: {result.best_score:.1%}
"""

from flowprompt import Prompt
from pydantic import BaseModel


class {optimized_class.__name__}(Prompt):
    """Optimized version of {prompt_class.__name__}."""

    system = """{getattr(optimized_class, "system", "")}"""
    user = """{getattr(optimized_class, "user", "")}"""
'''
                # Add Output class if present
                if hasattr(optimized_class, "Output"):
                    output_model = optimized_class.Output
                    if hasattr(output_model, "model_fields"):
                        fields = []
                        for fname, finfo in output_model.model_fields.items():
                            ftype = (
                                finfo.annotation.__name__
                                if hasattr(finfo.annotation, "__name__")
                                else str(finfo.annotation)
                            )
                            fields.append(f"        {fname}: {ftype}")
                        code += f"""
    class Output(BaseModel):
{chr(10).join(fields)}
"""

                output_path.write_text(code)
                typer.echo(f"\nOptimized prompt saved to: {output_path}")

        except Exception as e:
            typer.echo(f"Error during optimization: {e}", err=True)
            raise typer.Exit(1) from None

    @app.command("compare")
    def compare_cmd(
        variants_file: str = Argument(
            ..., help="Python file defining VARIANTS = {...} or Prompt classes"
        ),
        dataset: str = Argument(..., help="JSONL dataset, one example per line"),
        model: str = Option("gpt-4o", "--model", "-m", help="Default model"),
        metric: str = Option(
            "contains",
            "--metric",
            help="Scorer: exact, contains, regex, numeric, similarity",
        ),
        control: str = Option(None, "--control", help="Baseline variant name"),
        runs: int = Option(1, "--runs", help="Runs per input and variant"),
        confidence: float = Option(0.95, "--confidence", help="Confidence level"),
        temperature: float = Option(0.0, "--temperature", help="Temperature"),
        comparisons: str = Option(
            "control", "--comparisons", help="'control' or 'all' (pairs)"
        ),
        expected_key: str = Option("expected", "--expected-key"),
        input_key: str = Option("input", "--input-key"),
        variants_attr: str = Option("VARIANTS", "--variants-attr"),
        markdown: str = Option(
            None, "--markdown", help="Append a Markdown report to this file"
        ),
        html: str = Option(None, "--html", help="Write an HTML report"),
        json_out: str = Option(None, "--json", help="Write the result as JSON"),
        fail_on_regression: bool = Option(
            False,
            "--fail-on-regression",
            help="Exit 1 if a variant is significantly worse than the control",
        ),
    ) -> None:
        """Compare variants on a dataset with paired significance tests."""
        from flowprompt.cli.compare_cmd import run_compare_command

        code = run_compare_command(
            variants_file,
            dataset,
            model=model,
            metric=metric,
            control=control,
            runs=runs,
            confidence=confidence,
            temperature=temperature,
            comparisons=comparisons,
            expected_key=expected_key,
            input_key=input_key,
            variants_attr=variants_attr,
            markdown=markdown,
            html=html,
            json_path=json_out,
            fail_on_regression=fail_on_regression,
            echo=typer.echo,
        )
        if code:
            raise typer.Exit(code)

    return app


def main() -> int:
    """Main entry point for the FlowPrompt CLI."""
    if not HAS_TYPER:
        print("Error: Typer is required for the CLI.")
        print("Install it with: pip install flowprompt[dev] or pip install typer")
        return 1

    app = create_app()
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
