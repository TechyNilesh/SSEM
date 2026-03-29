"""Demo: Hallucination Detection Metric"""

from SSEM import SSEM

evaluator = SSEM()

output = "Paris is the capital of France. The Eiffel Tower is in London."
source = "Paris is the capital of France. The Eiffel Tower is located in Paris."

result = evaluator.hallucination(output, source)

print(f"Hallucination Score: {result.score:.4f}")
print(f"Hallucinated claims: {result.details['hallucinated_claims']}")
print()
print(result.explain())
