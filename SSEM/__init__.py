"""
SSEM — Standardized Scoring and Evaluation Metrics
====================================================

Lightweight evaluation metrics for LLMs and AI agents.
No platform. No API keys. Just scores.

Quick Start:
    from SSEM import SSEM

    evaluator = SSEM()
    result = evaluator.bertscore(
        ["The cat sat on the mat."],
        ["A cat was sitting on a mat."]
    )
    print(result.score)      # 0.87
    print(result.explain())  # Full transparency report

Available Metrics:
    - semantic_similarity  : Sentence-level semantic similarity
    - bertscore            : Token-level P/R/F1 (Zhang et al., 2020)
    - faithfulness         : NLI/embedding-based claim verification
    - hallucination        : Fraction of unsupported claims
    - answer_relevancy     : Question-answer relevance scoring
    - reasoning_coherence  : Multi-step reasoning chain evaluation
    - tool_accuracy        : AI agent tool call accuracy
    - task_completion      : Graded task completion scoring
    - multi_turn_consistency : Conversation consistency
    - selfcheck            : Sampling-based hallucination detection
    - toxicity             : Classifier-based toxicity scoring
    - code_correctness     : Execution-based Pass@k

Every metric returns a ``MetricResult`` with:
    - score, score_range, interpretation
    - method (step-by-step computation description)
    - model_used, citations, details, elapsed_sec
"""

__version__ = "2.0.0"

# Main entry point
from SSEM.evaluator import SSEM

# Core types (for advanced users)
from SSEM.core import (
    Citation,
    CITATIONS,
    EmbeddingEngine,
    MetricResult,
    BaseMetric,
)

# Individual metric classes (for direct use)
from SSEM.semantic import SemanticSimilarityMetric, BERTScoreMetric
from SSEM.faithfulness import FaithfulnessMetric, HallucinationMetric
from SSEM.relevancy import AnswerRelevancyMetric
from SSEM.agentic import (
    ReasoningCoherenceMetric,
    ToolCallAccuracyMetric,
    TaskCompletionMetric,
)
from SSEM.consistency import (
    MultiTurnConsistencyMetric,
    SelfCheckConsistencyMetric,
)
from SSEM.safety import ToxicityMetric
from SSEM.code_eval import CodeCorrectnessMetric
from SSEM.report import EvaluationReport

# Backward compatibility — original v1 class
from SSEM.SSEM import SemanticSimilarity

__all__ = [
    "SSEM",
    "__version__",
    # Core
    "Citation",
    "CITATIONS",
    "EmbeddingEngine",
    "MetricResult",
    "BaseMetric",
    "EvaluationReport",
    # Metrics
    "SemanticSimilarityMetric",
    "BERTScoreMetric",
    "FaithfulnessMetric",
    "HallucinationMetric",
    "AnswerRelevancyMetric",
    "ReasoningCoherenceMetric",
    "ToolCallAccuracyMetric",
    "TaskCompletionMetric",
    "MultiTurnConsistencyMetric",
    "SelfCheckConsistencyMetric",
    "ToxicityMetric",
    "CodeCorrectnessMetric",
    # Legacy
    "SemanticSimilarity",
]
