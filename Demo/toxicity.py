"""Demo: Toxicity Metric"""

from SSEM import SSEM

evaluator = SSEM()

result = evaluator.toxicity(
    texts=[
        "Have a wonderful day!",
        "This is a great library for evaluation.",
        "I hate everything about this.",
    ]
)

print(f"Mean Toxicity: {result.score:.4f}")
print()

for item in result.details["per_text"]:
    print(f"  [{item['toxicity_score']:.4f}] {item['text']}")
