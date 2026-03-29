# SSEM — Standardized Scoring and Evaluation Metrics

<p align="center">
  <img src="https://raw.githubusercontent.com/TechyNilesh/SSEM/main/assets/ssem-logo.png" alt="SSEM Logo" width="300">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/pypi/v/ssem.svg" alt="PyPI">
  <a href="https://pepy.tech/project/ssem"><img src="https://static.pepy.tech/personalized-badge/ssem?period=total&units=none&left_color=grey&right_color=green&left_text=Downloads" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center"><b>Lightweight evaluation metrics for LLMs and AI agents. No platform. No API keys. Just scores.</b></p>

SSEM provides 12 evaluation metrics covering text generation quality, factual consistency, hallucination detection, agentic AI evaluation, and safety — all with full scoring transparency and research citations.

## Installation

```bash
# PyPI
pip install ssem

# uv
uv pip install ssem

# Latest from GitHub
pip install git+https://github.com/TechyNilesh/SSEM.git
```

## Quick Start

```python
from SSEM import SSEM

evaluator = SSEM()

# BERTScore — one line
result = evaluator.bertscore(
    ["The cat sat on the mat."],
    ["A cat was sitting on a mat."]
)
print(result.score)      # 0.87
print(result.explain())  # Full transparency report
```

## Why SSEM?

| SSEM | DeepEval / Ragas |
|------|-----------------|
| **12 metrics in one lightweight package** | Bundled with platforms, tracing, dashboards |
| **No LLM-as-judge required** — embedding + NLI based | Often requires GPT-4 API calls ($$$) |
| **Agentic metrics built-in** — tool accuracy, reasoning chains | Focused on RAG, agents are afterthought |
| **Every score is transparent** — method, model, citations | Black-box scores |
| **Runs offline on CPU** — no API keys needed | Many require cloud API keys |

## Available Metrics

### Text Generation Quality

| Metric | Method | Score Range | Citation |
|--------|--------|-------------|----------|
| `semantic_similarity` | Sentence embedding cosine/euclidean/pearson similarity | [-1, 1] or [0, 1] | Vadapalli et al. (2021) |
| `bertscore` | Token-level precision, recall, F1 via contextual embeddings | [0, 1] | Zhang et al. (2020) |

### Factual Consistency

| Metric | Method | Score Range | Citation |
|--------|--------|-------------|----------|
| `faithfulness` | Claim extraction + NLI/embedding entailment checking | [0, 1] | Kryscinski et al. (2020) |
| `hallucination` | Fraction of output claims NOT grounded in source | [0, 1] | Kryscinski et al. (2020); Manakul et al. (2023) |
| `answer_relevancy` | Question-answer embedding similarity | [0, 1] | Es et al. (2024) |

### Agentic AI Evaluation

| Metric | Method | Score Range | Citation |
|--------|--------|-------------|----------|
| `reasoning_coherence` | Sequential + goal-aligned step similarity, contradiction detection | [0, 1] | Xia et al. (2024) |
| `tool_accuracy` | Tool selection + parameter + ordering accuracy (LCS) | [0, 1] | Liu et al. (2023) |
| `task_completion` | Checklist or reference-based graded completion | [0, 1] | Liu et al. (2023) |

### Consistency & Safety

| Metric | Method | Score Range | Citation |
|--------|--------|-------------|----------|
| `multi_turn_consistency` | Cross-turn semantic consistency + contradiction detection | [0, 1] | Zheng et al. (2023) |
| `selfcheck` | Sampling consistency for hallucination detection | [0, 1] | Manakul et al. (2023) |
| `toxicity` | Classifier-based toxicity scoring | [0, 1] | Gehman et al. (2020) |

### Code Evaluation

| Metric | Method | Score Range | Citation |
|--------|--------|-------------|----------|
| `code_correctness` | Execution-based Pass@k with unbiased estimator | [0, 1] | Chen et al. (2021) |

## Scoring Transparency

Every SSEM metric returns a `MetricResult` — never a bare number. Each result includes:

```python
result = evaluator.bertscore(outputs, references)

result.score           # 0.87 — the primary score
result.score_range     # (0.0, 1.0) — possible range
result.interpretation  # "Strong token-level overlap..."
result.method          # Step-by-step computation description
result.model_used      # "bert-base-multilingual-cased"
result.citations       # List of Citation objects
result.details         # Per-sample scores, intermediates
result.elapsed_sec     # Wall-clock time

# Full human-readable transparency report
print(result.explain())
```

Example output of `result.explain()`:

```
Metric        : BERTScore
Score         : 0.8734
Score Range   : [0.0, 1.0]
Interpretation: Strong token-level overlap — output captures most reference content.
Model Used    : bert-base-multilingual-cased

How This Score Was Computed:
1. Encoded 1 sentence pairs into per-token contextual embeddings using 'bert-base-multilingual-cased'.
2. For each pair, built a cosine similarity matrix between output tokens and reference tokens.
3. Precision = mean of row-wise max similarities (each output token's best reference match).
4. Recall = mean of column-wise max similarities (each reference token's best output match).
5. F1 = harmonic mean of precision and recall.
6. Averaged across 1 pairs: P=0.8912, R=0.8561, F1=0.8734.

Research Citations:
  [1] Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR 2020. https://arxiv.org/abs/1904.09675
```

## Usage Examples

### Text Generation Evaluation

```python
from SSEM import SSEM

evaluator = SSEM()

outputs = ["The cat sat on the mat.", "It was a sunny day."]
references = ["A cat was sitting on a mat.", "The weather was sunny."]

# Semantic similarity
result = evaluator.semantic_similarity(outputs, references)

# BERTScore (P/R/F1)
result = evaluator.bertscore(outputs, references)
print(result.details["precision"])  # 0.89
print(result.details["recall"])     # 0.85
print(result.details["f1"])         # 0.87
```

### Faithfulness & Hallucination

```python
output = "Paris is the capital of France. The Eiffel Tower is in London."
source = "Paris is the capital of France. The Eiffel Tower is located in Paris."

# Faithfulness — are claims grounded?
result = evaluator.faithfulness(output, source)
print(result.score)    # 0.5 — one of two claims is unfaithful
print(result.details)  # Per-claim breakdown with individual scores

# Hallucination — what fraction is fabricated?
result = evaluator.hallucination(output, source)
print(result.score)    # 0.5 — half the claims are hallucinated
```

### Agentic AI Evaluation

```python
# Reasoning chain coherence
result = evaluator.reasoning_coherence(
    reasoning_steps=[
        "First, I need to find the user's order history.",
        "Next, I'll filter orders from the last 30 days.",
        "Then, I'll calculate the total spending.",
        "Finally, I'll generate a summary report.",
    ],
    goal="Generate a spending report for the last month."
)
print(result.score)                       # 0.82
print(result.details["contradictions"])   # [] — no contradictions

# Tool call accuracy
result = evaluator.tool_accuracy(
    predicted_calls=[
        {"tool": "database_query", "params": {"table": "orders", "days": 30}},
        {"tool": "calculate_sum", "params": {"column": "amount"}},
    ],
    expected_calls=[
        {"tool": "database_query", "params": {"table": "orders", "days": 30}},
        {"tool": "calculate_sum", "params": {"column": "amount"}},
    ],
)
print(result.score)  # 1.0 — perfect tool usage

# Task completion (checklist mode)
result = evaluator.task_completion(
    agent_output="I queried the database and found 15 orders totaling $1,234.",
    expected_criteria=[
        "Query the order database",
        "Calculate total spending",
        "Report the number of orders",
    ],
)
print(result.score)  # 0.67 — 2 of 3 criteria met
```

### Multi-Turn Consistency

```python
result = evaluator.multi_turn_consistency(
    responses=[
        "I recommend Python for this project.",
        "Python has great ML libraries like scikit-learn.",
        "Actually, you should use Java instead.",  # contradiction!
    ]
)
print(result.score)                     # 0.61
print(result.details["contradictions"]) # Flags the Java contradiction
```

### Code Correctness

```python
result = evaluator.code_correctness(
    code_samples=[
        "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)",
        "def factorial(n):\n    return n * n",  # wrong
    ],
    test_code="assert factorial(5) == 120\nassert factorial(0) == 1",
    k_values=[1, 2],
)
print(result.details["pass_at_k"])  # {"pass@1": 0.5, "pass@2": 1.0}
```

### Full Evaluation Report

```python
report = evaluator.evaluate_all(
    output_sentences=["The cat sat on the mat."],
    reference_sentences=["A cat was sitting on a mat."],
    source_context="A cat was observed sitting on a mat in the room.",
    reasoning_steps=["Find the cat.", "Describe its position."],
)

print(report.summary())   # One-line-per-metric table
print(report.explain())   # Full transparency + bibliography
print(report.to_json())   # JSON export for pipelines
```

## Research Citations

SSEM is grounded in peer-reviewed research. Every metric cites its origin:

| Metric | Paper | Venue |
|--------|-------|-------|
| BERTScore | Zhang et al. "BERTScore: Evaluating Text Generation with BERT" | ICLR 2020 |
| Semantic Similarity | Beken Fikri et al. "Semantic Similarity Based Evaluation for Abstractive News Summarization" | GEM @ ACL 2021 |
| Faithfulness | Kryscinski et al. "Evaluating the Factual Consistency of Abstractive Text Summarization" | EMNLP 2020 |
| SelfCheck | Manakul et al. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection" | EMNLP 2023 |
| Answer Relevancy | Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation" | EACL 2024 |
| Reasoning Coherence | Xia, Li, Liu, Wu & Liu. "ReasonEval: Evaluating Mathematical Reasoning Beyond Accuracy" | arXiv 2024 |
| AgentBench | Liu et al. "AgentBench: Evaluating LLMs as Agents" | ICLR 2024 |
| Pass@k | Chen et al. "Evaluating Large Language Models Trained on Code" | arXiv 2021 |
| MT-Bench | Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" | NeurIPS 2023 |
| Toxicity | Gehman et al. "RealToxicityPrompts" | EMNLP 2020 |
| LSI | Deerwester et al. "Indexing by Latent Semantic Analysis" | JASIS 1990 |
| NLI | Williams et al. "MultiNLI" | NAACL 2018 |

Access all citations programmatically:

```python
evaluator.list_citations()  # Returns dict of all citations
```

## Architecture

```
SSEM/
├── __init__.py        # Package exports
├── evaluator.py       # SSEM unified class — main entry point
├── core.py            # EmbeddingEngine, MetricResult, Citation, BaseMetric
├── semantic.py        # SemanticSimilarity, BERTScore
├── faithfulness.py    # Faithfulness, Hallucination
├── relevancy.py       # AnswerRelevancy
├── agentic.py         # ReasoningCoherence, ToolCallAccuracy, TaskCompletion
├── consistency.py     # MultiTurnConsistency, SelfCheckConsistency
├── safety.py          # Toxicity
├── code_eval.py       # CodeCorrectness (Pass@k)
├── report.py          # EvaluationReport
└── SSEM.py            # Legacy v1 (backward compatible)
```

## Parameters

### SSEM Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"bert-base-multilingual-cased"` | Any HuggingFace model |
| `device` | str | Auto-detected | `"cpu"`, `"cuda"`, or `"mps"` |

### Common Method Parameters

All methods accept the specific inputs for their metric and return a `MetricResult`.

See `evaluator.list_metrics()` for all available metrics.

## Backward Compatibility

The original v1 API still works:

```python
from SSEM import SemanticSimilarity

ssem = SemanticSimilarity(model_name='bert-base-multilingual-cased', metric='cosine')
score = ssem.evaluate(output_sentences, reference_sentences, level='sentence', output_format='mean')
```

## Core Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/TechyNilesh">
        <img src="https://github.com/TechyNilesh.png" width="100px;" alt="Nilesh Verma"/>
        <br />
        <b>Nilesh Verma</b>
      </a>
      <br />
      <a href="https://nileshverma.com">Website</a> · <a href="https://github.com/TechyNilesh">GitHub</a>
    </td>
  </tr>
</table>

## Citation

If you use `SSEM` in your research, please cite:

```bibtex
@misc{SSEM,
  author = {Nilesh Verma},
  title = {SSEM: Standardized Scoring and Evaluation Metrics},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TechyNilesh/SSEM}}
}
```

## License

SSEM is released under the MIT License.
