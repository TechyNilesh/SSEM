"""Demo: BERTScore Metric"""

from SSEM import SSEM

evaluator = SSEM()

output_sentences = ["The cat sat on the mat.", "It was a sunny day."]
reference_sentences = ["A cat was sitting on a mat.", "The weather was sunny."]

result = evaluator.bertscore(output_sentences, reference_sentences)

print(f"F1:        {result.score:.4f}")
print(f"Precision: {result.details['precision']:.4f}")
print(f"Recall:    {result.details['recall']:.4f}")
print()
print(result.explain())
