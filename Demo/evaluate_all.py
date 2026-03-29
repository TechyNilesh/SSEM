"""Demo: Full Evaluation Report (evaluate_all)"""

from SSEM import SSEM

evaluator = SSEM()

report = evaluator.evaluate_all(
    output_sentences=["The cat sat on the mat."],
    reference_sentences=["A cat was sitting on a mat."],
    source_context="A cat was observed sitting on a mat in the room.",
    reasoning_steps=["Find the cat.", "Describe its position."],
    conversation_turns=[
        "The cat is on the mat.",
        "Yes, the cat is sitting on the mat.",
    ],
)

# Summary table
print(report.summary())
print()

# Full transparency report with bibliography
print(report.explain())
print()

# JSON export
print(report.to_json())
