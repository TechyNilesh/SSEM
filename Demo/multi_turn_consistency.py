"""Demo: Multi-Turn Consistency Metric"""

from SSEM import SSEM

evaluator = SSEM()

result = evaluator.multi_turn_consistency(
    responses=[
        "I recommend Python for this project.",
        "Python has great ML libraries like scikit-learn.",
        "Actually, you should use Java instead.",  # contradiction
    ]
)

print(f"Consistency Score: {result.score:.4f}")
print(f"Sequential: {result.details['sequential_consistency']:.4f}")
print(f"Global:     {result.details['global_consistency']:.4f}")
print()

if result.details["contradictions"]:
    print("Contradictions detected:")
    for c in result.details["contradictions"]:
        print(f"  Turn {c['turn_pair'][0]} vs {c['turn_pair'][1]} (sim={c['similarity']:.4f})")
