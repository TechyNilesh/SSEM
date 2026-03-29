"""Demo: Semantic Similarity Metric"""

from SSEM import SSEM

evaluator = SSEM()

output_sentences = ["The cat sat on the mat.", "It was a sunny day."]
reference_sentences = ["A cat was sitting on a mat.", "The weather was sunny."]

# Cosine similarity (default)
result = evaluator.semantic_similarity(output_sentences, reference_sentences)
print(result.score)
print(result.explain())

# Euclidean similarity
result = evaluator.semantic_similarity(output_sentences, reference_sentences, metric="euclidean")
print(result.score)

# Pearson similarity
result = evaluator.semantic_similarity(output_sentences, reference_sentences, metric="pearson")
print(result.score)
