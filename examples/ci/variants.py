from flowprompt import Prompt


class Current(Prompt):
    system = (
        "Classify the sentiment. Reply with one word: positive, negative or neutral."
    )
    user = "Review: {text}"


class Candidate(Prompt):
    system = "You are a friendly assistant. What is the sentiment of this review?"
    user = "Review: {text}"


VARIANTS = {"current": Current, "candidate": Candidate}
