"""Demo: Answer Relevancy Metric"""

from SSEM import SSEM

evaluator = SSEM()

questions = [
    "What is the capital of France?",
    "How does photosynthesis work?",
]
answers = [
    "Paris is the capital of France.",
    "I like pizza.",  # irrelevant answer
]

result = evaluator.answer_relevancy(questions, answers)

print(f"Mean Relevancy: {result.score:.4f}")
print()
for pair in result.details["per_pair"]:
    print(f"  Q: {pair['question']}")
    print(f"  A: {pair['answer']}")
    print(f"  Relevancy: {pair['relevancy']:.4f}")
    print()
