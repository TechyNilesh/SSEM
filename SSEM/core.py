"""
SSEM Core Engine
================
Provides the base infrastructure for all evaluation metrics:
- EmbeddingEngine: Unified model loading and sentence encoding
- MetricResult: Transparent, self-documenting evaluation result
- Citation: Research paper reference attached to every metric
- BaseMetric: Abstract base class enforcing citation + transparency

Design Principles:
    1. Every score must be explainable — no black-box numbers
    2. Every metric must cite its research origin
    3. Every result includes the method, model, and scoring range
"""

from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Citation Registry
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Citation:
    """Immutable reference to a research paper.

    Every metric in SSEM is grounded in published research.  Attaching a
    ``Citation`` makes the provenance of each score auditable.
    """

    title: str
    authors: str
    year: int
    venue: str  # e.g. "ICLR 2020", "ACL 2021", "arXiv"
    url: str
    description: str  # one-line summary of what this paper contributes

    def __str__(self) -> str:
        return f"{self.authors} ({self.year}). {self.title}. {self.venue}. {self.url}"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# Pre-defined citations used across metrics
CITATIONS: Dict[str, Citation] = {
    "bertscore": Citation(
        title="BERTScore: Evaluating Text Generation with BERT",
        authors="Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y.",
        year=2020,
        venue="ICLR 2020",
        url="https://arxiv.org/abs/1904.09675",
        description="Token-level semantic similarity using contextual embeddings to compute precision, recall, and F1.",
    ),
    "semantic_similarity": Citation(
        title="Semantic Similarity Based Evaluation for Abstractive News Summarization",
        authors="Beken Fikri, F., Oflazer, K., & Yanıkoğlu, B.",
        year=2021,
        venue="Proceedings of the First Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2021)",
        url="https://aclanthology.org/2021.gem-1.3/",
        description="Semantic similarity between transformer embeddings as a generation quality metric for abstractive summarization.",
    ),
    "nli_faithfulness": Citation(
        title="Evaluating the Factual Consistency of Abstractive Text Summarization",
        authors="Kryscinski, W., McCann, B., Xiong, C., & Socher, R.",
        year=2020,
        venue="EMNLP 2020",
        url="https://arxiv.org/abs/1910.12840",
        description="NLI-based factual consistency checking: decompose output into claims and verify entailment against source.",
    ),
    "selfcheckgpt": Citation(
        title="SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models",
        authors="Manakul, P., Liusie, A., & Gales, M. J. F.",
        year=2023,
        venue="EMNLP 2023",
        url="https://arxiv.org/abs/2303.08896",
        description="Detect hallucinations by sampling multiple responses and measuring inter-sample consistency.",
    ),
    "answer_relevancy": Citation(
        title="RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        authors="Es, S., James, J., Espinosa-Anke, L., & Schockaert, S.",
        year=2024,
        venue="EACL 2024",
        url="https://arxiv.org/abs/2309.15217",
        description="Embedding-based answer relevancy: measure how well the answer addresses the question.",
    ),
    "reasoning_coherence": Citation(
        title="ReasonEval: Evaluating Mathematical Reasoning Beyond Accuracy",
        authors="Xia, S., Li, X., Liu, Y., Wu, T., & Liu, P.",
        year=2024,
        venue="arXiv",
        url="https://arxiv.org/abs/2404.05692",
        description="Step-level evaluation of reasoning chains for validity and logical coherence.",
    ),
    "agent_eval": Citation(
        title="AgentBench: Evaluating LLMs as Agents",
        authors="Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., et al.",
        year=2023,
        venue="ICLR 2024",
        url="https://arxiv.org/abs/2308.03688",
        description="Multi-dimensional benchmark for evaluating LLM agents on task completion, tool use, and planning.",
    ),
    "pass_at_k": Citation(
        title="Evaluating Large Language Models Trained on Code",
        authors="Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. de O., et al.",
        year=2021,
        venue="arXiv",
        url="https://arxiv.org/abs/2107.03374",
        description="Pass@k: unbiased estimator for functional code correctness via repeated sampling and execution.",
    ),
    "multi_turn": Citation(
        title="Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
        authors="Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., et al.",
        year=2023,
        venue="NeurIPS 2023",
        url="https://arxiv.org/abs/2306.05685",
        description="MT-Bench: multi-turn benchmark measuring consistency, instruction following, and knowledge retention.",
    ),
    "toxicity": Citation(
        title="RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models",
        authors="Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N. A.",
        year=2020,
        venue="Findings of EMNLP 2020",
        url="https://arxiv.org/abs/2009.11462",
        description="Framework for measuring toxicity in language model outputs using classifier-based scoring.",
    ),
    "lsi": Citation(
        title="Indexing by Latent Semantic Analysis",
        authors="Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R.",
        year=1990,
        venue="Journal of the American Society for Information Science",
        url="https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9",
        description="Latent Semantic Indexing: dimensionality reduction on term-document matrices for semantic similarity.",
    ),
    "cosine_similarity": Citation(
        title="Introduction to Information Retrieval",
        authors="Manning, C. D., Raghavan, P., & Schütze, H.",
        year=2008,
        venue="Cambridge University Press",
        url="https://nlp.stanford.edu/IR-book/",
        description="Cosine similarity as a standard vector-space similarity measure for text representations.",
    ),
    "nli_models": Citation(
        title="A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference",
        authors="Williams, A., Nangia, N., & Bowman, S.",
        year=2018,
        venue="NAACL 2018",
        url="https://arxiv.org/abs/1704.05426",
        description="MultiNLI corpus and NLI task definition used as basis for entailment-based factual verification.",
    ),
}


# ---------------------------------------------------------------------------
# Transparent Metric Result
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MetricResult:
    """Every SSEM metric returns this — never a bare float.

    Attributes:
        score:        The primary numeric score.
        metric_name:  Human-readable metric name (e.g. "BERTScore F1").
        score_range:  Tuple (min, max) describing the possible range.
        interpretation: What the score means in plain English.
        method:       Step-by-step description of how the score was computed.
        model_used:   Which model produced the embeddings / predictions.
        citations:    List of Citation objects grounding this metric.
        details:      Optional dict with per-sample scores, intermediates, etc.
        elapsed_sec:  Wall-clock time for this evaluation.
    """

    score: float
    metric_name: str
    score_range: Tuple[float, float]
    interpretation: str
    method: str
    model_used: str
    citations: List[Citation]
    details: Optional[Dict[str, Any]] = None
    elapsed_sec: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"MetricResult(metric={self.metric_name!r}, score={self.score:.4f}, "
            f"range={self.score_range}, model={self.model_used!r})"
        )

    def explain(self) -> str:
        """Return a full human-readable transparency report."""
        lines = [
            f"Metric        : {self.metric_name}",
            f"Score         : {self.score:.4f}",
            f"Score Range   : [{self.score_range[0]}, {self.score_range[1]}]",
            f"Interpretation: {self.interpretation}",
            f"Model Used    : {self.model_used}",
            "",
            "How This Score Was Computed:",
            self.method,
            "",
            "Research Citations:",
        ]
        for i, c in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {c}")
        if self.elapsed_sec is not None:
            lines.append(f"\nComputed in {self.elapsed_sec:.3f}s")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "metric_name": self.metric_name,
            "score_range": list(self.score_range),
            "interpretation": self.interpretation,
            "method": self.method,
            "model_used": self.model_used,
            "citations": [c.to_dict() for c in self.citations],
            "details": self.details,
            "elapsed_sec": self.elapsed_sec,
        }


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Unified encoder for all SSEM metrics that need embeddings.

    Loads a HuggingFace transformer once and provides:
    - ``encode()``: mean-pooled sentence embeddings
    - ``encode_cls()``: CLS-token embeddings (legacy SSEM behaviour)
    - ``encode_tokens()``: per-token embeddings for BERTScore-style metrics

    Parameters:
        model_name: Any HuggingFace model identifier.
        device:     "cpu", "cuda", or "mps".  Auto-detected if None.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """Mean-pooled sentence embeddings — best general-purpose representation.

        Method:
            1. Tokenize with padding and truncation.
            2. Forward pass through the transformer.
            3. Mean-pool the last hidden states (excluding padding tokens).
        """
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            outputs = self.model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = outputs.last_hidden_state
            # Mean pooling: sum token embeddings weighted by attention mask
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            all_embeddings.append(pooled.cpu().numpy())
        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_cls(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """CLS-token embeddings — legacy SSEM v1 behaviour."""
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_emb.cpu().numpy())
        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_tokens(
        self, sentences: List[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Per-token embeddings for each sentence (variable-length).

        Returns a list of arrays, one per sentence, shape (num_tokens, hidden_dim).
        Padding tokens are excluded.
        """
        all_token_embs: List[np.ndarray] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state.cpu().numpy()
            mask = inputs["attention_mask"].cpu().numpy()
            for j in range(len(batch)):
                length = int(mask[j].sum())
                all_token_embs.append(hidden[j, :length, :])
        return all_token_embs

    def __repr__(self) -> str:
        return f"EmbeddingEngine(model={self.model_name!r}, device={self.device!r})"


# ---------------------------------------------------------------------------
# Base Metric (abstract)
# ---------------------------------------------------------------------------

class BaseMetric(ABC):
    """Abstract base class that every SSEM metric must implement.

    Enforces two guarantees:
        1. ``citations`` — every metric declares its research provenance.
        2. ``evaluate()`` — returns a ``MetricResult``, never a bare number.
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Human-readable name shown in reports."""
        ...

    @property
    @abstractmethod
    def citations(self) -> List[Citation]:
        """Research papers this metric is based on."""
        ...

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> MetricResult:
        """Run the evaluation and return a transparent result."""
        ...

    def _timed(self, fn, *args, **kwargs) -> Tuple[Any, float]:
        """Helper to time a callable and return (result, elapsed_seconds)."""
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
