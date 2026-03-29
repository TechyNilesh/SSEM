"""Demo: SelfCheck Consistency Metric"""

from SSEM import SSEM

evaluator = SSEM()

result = evaluator.selfcheck(
    main_response="Paris is the capital of France. It has a population of 2.1 million.",
    sampled_responses=[
        "The capital of France is Paris. Its population is about 2.1 million people.",
        "Paris serves as France's capital city. Around 2.1 million people live there.",
        "France's capital is Paris, home to roughly 2 million residents.",
    ],
)

print(f"Inconsistency Score: {result.score:.4f}")
print(f"(0 = consistent, 1 = inconsistent)")
print()

for claim in result.details["claim_results"]:
    status = "Consistent" if claim["consistent"] else "INCONSISTENT"
    print(f"  [{status}] {claim['claim']} (avg_sim={claim['avg_similarity_to_samples']:.4f})")
