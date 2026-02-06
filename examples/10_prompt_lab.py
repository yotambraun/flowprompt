"""Prompt Lab: Compare prompt variants with statistical testing.

This example shows how to use FlowPrompt's compare() function to
find the best prompt variant for a sentiment analysis task.

The compare() function runs each variant against your test inputs,
measures success rates and latency, and performs a statistical
significance test to determine the winner.

Requirements:
    pip install flowprompt-ai

    Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
"""

import os

from flowprompt import Prompt, compare


# =============================================================================
# Step 1: Define prompt variants to compare
# =============================================================================


class SentimentBasic(Prompt):
    """Simple, minimal prompt."""

    system = "Classify sentiment."
    user = "{text}"


class SentimentDetailed(Prompt):
    """More detailed instructions."""

    system = (
        "You are a sentiment analysis expert. "
        "Classify the sentiment of the text as exactly one of: "
        "positive, negative, or neutral. Reply with a single word."
    )
    user = "Text: {text}"


class SentimentFewShot(Prompt):
    """Few-shot prompt with examples baked in."""

    system = (
        "Classify text sentiment as positive, negative, or neutral.\n\n"
        "Examples:\n"
        '"I love this!" -> positive\n'
        '"This is terrible." -> negative\n'
        '"The meeting is at 3pm." -> neutral\n\n'
        "Reply with a single word."
    )
    user = "Text: {text}"


# =============================================================================
# Step 2: Define test inputs and success criteria
# =============================================================================

TEST_INPUTS = [
    {"text": "I absolutely love this product!"},
    {"text": "Worst experience I've ever had."},
    {"text": "The package arrived on Tuesday."},
    {"text": "This is amazing, best purchase ever!"},
    {"text": "I'm so frustrated with the service."},
    {"text": "The report contains 50 pages."},
    {"text": "What a wonderful surprise!"},
    {"text": "I regret buying this."},
    {"text": "The temperature today is 72F."},
    {"text": "Exceeded all my expectations!"},
]

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def is_valid_sentiment(output: str) -> bool:
    """Check if the output is a valid single-word sentiment."""
    return output.strip().lower() in VALID_SENTIMENTS


# =============================================================================
# Step 3: Preview messages (no API key needed)
# =============================================================================


def preview_prompts() -> None:
    """Show what each variant sends to the LLM."""
    print("Prompt Variants Preview")
    print("=" * 50)

    sample = TEST_INPUTS[0]

    for name, cls in [
        ("basic", SentimentBasic),
        ("detailed", SentimentDetailed),
        ("few_shot", SentimentFewShot),
    ]:
        prompt = cls(**sample)
        messages = prompt.to_messages()
        print(f"\n--- {name} ---")
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 80:
                content = content[:77] + "..."
            print(f"  [{role}] {content}")


# =============================================================================
# Step 4: Run the comparison (requires API key)
# =============================================================================


def run_comparison() -> None:
    """Run all three variants and print results."""
    print("\nRunning Comparison")
    print("=" * 50)

    result = compare(
        {
            "basic": SentimentBasic,
            "detailed": SentimentDetailed,
            "few_shot": SentimentFewShot,
        },
        inputs=TEST_INPUTS,
        model="gpt-4o-mini",
        success_fn=is_valid_sentiment,
        temperature=0.0,
    )

    print(result)
    print()

    # Access programmatic results
    for name, variant in result.variants.items():
        print(f"{name}:")
        print(f"  Success rate: {variant.success_rate:.0%}")
        print(f"  Avg latency:  {variant.mean_latency_ms:.0f}ms")
        print(f"  Errors:       {len(variant.errors)}")

    if result.winner:
        print(f"\nRecommendation: Use '{result.winner}' in production.")
    else:
        print("\nNo statistically significant winner. Try more inputs or runs_per_input.")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the prompt lab example."""
    print("FlowPrompt Prompt Lab")
    print("Find the best prompt variant with statistical testing")
    print()

    # Always show the preview (no API key needed)
    preview_prompts()
    print()

    # Check for API key before running comparison
    has_key = any(
        os.environ.get(k)
        for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    )

    if has_key:
        run_comparison()
    else:
        print("Set OPENAI_API_KEY (or another provider key) to run the comparison.")
        print("The preview above works without any API key.")


if __name__ == "__main__":
    main()
