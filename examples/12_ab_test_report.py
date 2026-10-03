"""A/B test two prompts and get a report you can trust (and share).

Two sentiment prompts are compared on the same 24 labelled reviews.
compare() runs both on every review, grades each answer against the label,
and tests the difference with a *paired* test (exact McNemar), because both
prompts saw exactly the same inputs.

Without an API key the script runs offline: a simulated model answers
(the concise prompt replies with one word; the chatty prompt often wraps
the label in a sentence, which an exact-match grader rejects). With a key it
calls the real model.

Run:
    python examples/12_ab_test_report.py
Writes ab_test_report.html (self-contained) and ab_test_report.md.
"""

import os
import time

from flowprompt import Prompt, compare
from flowprompt.testing import FakeLLM


class Concise(Prompt):
    system = (
        "Classify the sentiment. Reply with one word: positive, negative or neutral."
    )
    user = "Review: {text}"


class Chatty(Prompt):
    system = "You are a helpful assistant. Analyze the sentiment of the review."
    user = "Review: {text}"


REVIEWS = [
    ("Absolutely love it, works perfectly.", "positive"),
    ("Broke after two days. Waste of money.", "negative"),
    ("It arrived on Tuesday.", "neutral"),
    ("Best purchase I made this year!", "positive"),
    ("The battery life is awful.", "negative"),
    ("Comes in a blue box.", "neutral"),
    ("Great value and fast shipping.", "positive"),
    ("Customer support never answered.", "negative"),
    ("The manual is twelve pages long.", "neutral"),
    ("Oh great, another charger that melts.", "negative"),
    ("Exceeded my expectations.", "positive"),
    ("Stopped working, returning it.", "negative"),
    ("Available in three sizes.", "neutral"),
    ("My kids adore it.", "positive"),
    ("Cheap plastic, feels flimsy.", "negative"),
    ("Ships from Germany.", "neutral"),
    ("Five stars, would buy again.", "positive"),
    ("Worst headphones I have owned.", "negative"),
    ("The cable is 1.5 meters.", "neutral"),
    ("Wow, it only took three weeks to arrive. Impressive.", "negative"),
    ("Sturdy, quiet and easy to clean.", "positive"),
    ("Screen cracked on day one.", "negative"),
    ("Weighs about 300 grams.", "neutral"),
    ("Exactly what I needed.", "positive"),
]


def simulated_model(messages: list[dict]) -> str:
    """Offline stand-in for an LLM, used only when no API key is set."""
    system, review = messages[0]["content"], messages[-1]["content"]
    text = review.removeprefix("Review: ")
    label = dict(REVIEWS)[text]
    index = [r for r, _ in REVIEWS].index(text)
    sarcastic = text.startswith(("Oh great", "Wow"))
    if "one word" in system:
        answer = "positive" if sarcastic else label
    elif index % 3 == 0:
        answer = label
    else:
        # The chatty prompt is right, but often answers in a full sentence.
        answer = f"The sentiment of this review is {label}."
    time.sleep(0.004 * len(answer) ** 0.5)  # longer answers take longer
    return answer


def main() -> None:
    has_key = any(
        os.environ.get(k)
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")
    ) and not os.environ.get("OPENAI_API_KEY", "").startswith("sk-placeholder")

    def run():
        return compare(
            {"concise": Concise, "chatty": Chatty},
            inputs=[{"text": text} for text, _ in REVIEWS],
            expected=[label for _, label in REVIEWS],
            eval_metric="exact",
            model="gpt-4o-mini",
        )

    if has_key:
        result = run()
    else:
        print("(No API key found: answering with a simulated model, offline.)\n")
        with FakeLLM(simulated_model):
            result = run()

    print(result)
    print()
    result.save_report("ab_test_report.html")
    result.save_report("ab_test_report.md")
    print("Wrote ab_test_report.html and ab_test_report.md")


if __name__ == "__main__":
    main()
