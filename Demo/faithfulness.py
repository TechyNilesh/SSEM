"""Demo: Faithfulness Metric"""

from SSEM import SSEM

evaluator = SSEM()

output = "Paris is the capital of France. The Eiffel Tower is in London."
source = "Paris is the capital of France. The Eiffel Tower is located in Paris."

result = evaluator.faithfulness(output, source)

print(f"Score: {result.score:.4f}")
print()

# Per-claim breakdown
for claim in result.details["claim_results"]:
    status = "Faithful" if claim["faithful"] else "NOT faithful"
    print(f"  [{status}] {claim['claim']}")

print()
print(result.explain())
