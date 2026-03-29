"""Demo: Code Correctness (Pass@k) Metric"""

from SSEM import SSEM

evaluator = SSEM()

result = evaluator.code_correctness(
    code_samples=[
        "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)",
        "def factorial(n):\n    return n * n",  # wrong
        "def factorial(n):\n    result = 1\n    for i in range(2, n+1): result *= i\n    return result",
    ],
    test_code="assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(1) == 1",
    k_values=[1, 3],
)

print(f"Pass@1: {result.details['pass_at_k']['pass@1']:.4f}")
print(f"Pass@3: {result.details['pass_at_k']['pass@3']:.4f}")
print(f"Passed: {result.details['pass_count']}/{result.details['total_samples']}")
print()

for s in result.details["sample_results"]:
    status = "PASS" if s["passed"] else "FAIL"
    print(f"  Sample {s['sample_index']}: {status}")
    if s["error"]:
        print(f"    Error: {s['error'][:100]}")
