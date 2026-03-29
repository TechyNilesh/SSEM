"""Demo: Reasoning Coherence Metric"""

from SSEM import SSEM

evaluator = SSEM()

result = evaluator.reasoning_coherence(
    reasoning_steps=[
        "First, I need to find the user's order history.",
        "Next, I'll filter orders from the last 30 days.",
        "Then, I'll calculate the total spending.",
        "Finally, I'll generate a summary report.",
    ],
    goal="Generate a spending report for the last month.",
)

print(f"Coherence Score: {result.score:.4f}")
print(f"Sequential Coherence: {result.details['sequential_coherence']:.4f}")
print(f"Goal Alignment: {result.details['goal_alignment']:.4f}")
print(f"Contradictions: {len(result.details['contradictions'])}")
print()
print(result.explain())
