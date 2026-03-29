"""Demo: Tool Call Accuracy Metric"""

from SSEM import SSEM

evaluator = SSEM()

predicted = [
    {"tool": "database_query", "params": {"table": "orders", "days": 30}},
    {"tool": "calculate_sum", "params": {"column": "amount"}},
    {"tool": "send_email", "params": {"to": "user@example.com"}},
]

expected = [
    {"tool": "database_query", "params": {"table": "orders", "days": 30}},
    {"tool": "calculate_sum", "params": {"column": "total"}},  # wrong param
]

result = evaluator.tool_accuracy(predicted, expected)

print(f"Tool Accuracy: {result.score:.4f}")
print(f"  Selection: {result.details['selection_accuracy']:.4f}")
print(f"  Params:    {result.details['param_accuracy']:.4f}")
print(f"  Ordering:  {result.details['order_accuracy']:.4f}")
print()
print(result.explain())
