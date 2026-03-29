"""Demo: Task Completion Metric"""

from SSEM import SSEM

evaluator = SSEM()

# Checklist mode
result = evaluator.task_completion(
    agent_output="I queried the database and found 15 orders totaling $1,234.",
    expected_criteria=[
        "Query the order database",
        "Calculate total spending",
        "Report the number of orders",
    ],
)

print(f"Task Completion: {result.score:.4f}")
for cr in result.details["criteria_results"]:
    status = "MET" if cr["achieved"] else "NOT MET"
    print(f"  [{status}] {cr['criterion']} (sim={cr['similarity']:.4f})")

print()

# Reference mode
result = evaluator.task_completion(
    agent_output="The analysis is complete with 15 orders.",
    reference_output="The analysis has been completed successfully with all orders counted.",
)

print(f"Reference Completion: {result.score:.4f}")
