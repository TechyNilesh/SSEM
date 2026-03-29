"""
Agentic AI Evaluation Metrics
=============================

Metrics designed specifically for evaluating AI agents — systems that
plan, reason, use tools, and complete multi-step tasks.

1. **ReasoningCoherenceMetric** — Evaluates logical consistency across
   a chain of reasoning steps.
2. **ToolCallAccuracyMetric** — Measures whether an agent selected the
   correct tools with correct parameters.
3. **TaskCompletionMetric** — Graded (not just pass/fail) evaluation of
   how completely an agent achieved its goal.

Research Basis:
    - Xia, S., Li, X., Liu, Y., Wu, T., & Liu, P. (2024). ReasonEval.
    - Liu et al. (2023). AgentBench: LLM agent evaluation. ICLR 2024.
    - Chen et al. (2021). Evaluating LLMs Trained on Code. (Pass@k concept)

These are the metrics that are most lacking in the current evaluation
ecosystem. Existing tools (DeepEval, Ragas) focus on RAG — agent
evaluation is an open problem.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from SSEM.core import (
    BaseMetric,
    Citation,
    CITATIONS,
    EmbeddingEngine,
    MetricResult,
)


class ReasoningCoherenceMetric(BaseMetric):
    """Evaluates the logical coherence of a multi-step reasoning chain.

    Research Basis:
        Xia, S., Li, X., Liu, Y., Wu, T., & Liu, P. (2024).
        ReasonEval: Evaluating Mathematical Reasoning Beyond Accuracy.
        https://arxiv.org/abs/2404.05692

    How Scoring Works:
        1. Each reasoning step is encoded into a dense embedding.
        2. **Sequential Coherence**: Cosine similarity between consecutive
           steps (step_i, step_{i+1}). A coherent chain should have high
           similarity between adjacent steps.
        3. **Goal Alignment**: If a goal/question is provided, each step's
           similarity to the goal is measured. Steps should remain
           aligned with the objective.
        4. **Contradiction Detection**: Large drops in sequential similarity
           (below a threshold) flag potential logical contradictions.
        5. **Final Score** = weighted combination of sequential coherence
           and goal alignment.

    Score Range: [0.0, 1.0] where 1.0 = perfectly coherent reasoning chain.

    Parameters:
        engine:              EmbeddingEngine for step encoding.
        contradiction_threshold: Similarity drop flagged as contradiction.
        goal_weight:         Weight of goal alignment in final score (0-1).
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        contradiction_threshold: float = 0.3,
        goal_weight: float = 0.3,
    ) -> None:
        self.engine = engine
        self.contradiction_threshold = contradiction_threshold
        self.goal_weight = goal_weight

    @property
    def metric_name(self) -> str:
        return "Reasoning Coherence"

    @property
    def citations(self) -> List[Citation]:
        return [CITATIONS["reasoning_coherence"], CITATIONS["agent_eval"]]

    def evaluate(
        self,
        reasoning_steps: List[str],
        goal: Optional[str] = None,
    ) -> MetricResult:
        """Evaluate coherence of a reasoning chain.

        Args:
            reasoning_steps: Ordered list of reasoning steps (strings).
            goal:            Optional goal/question the reasoning should address.

        Returns:
            MetricResult with per-step coherence and contradiction flags.
        """
        if len(reasoning_steps) < 2:
            return MetricResult(
                score=1.0,
                metric_name=self.metric_name,
                score_range=(0.0, 1.0),
                interpretation="Single step — coherence is trivially 1.0.",
                method="Only one reasoning step provided; no chain to evaluate.",
                model_used=self.engine.model_name,
                citations=self.citations,
            )

        start = time.perf_counter()

        step_embs = self.engine.encode(reasoning_steps)

        # Sequential coherence: cosine sim between consecutive steps
        sequential_sims = []
        for i in range(len(step_embs) - 1):
            sim = cosine_similarity(
                step_embs[i : i + 1], step_embs[i + 1 : i + 2]
            )[0, 0]
            sequential_sims.append(float(sim))

        seq_coherence = float(np.mean(sequential_sims))

        # Contradiction detection
        contradictions = []
        for i, sim in enumerate(sequential_sims):
            if sim < self.contradiction_threshold:
                contradictions.append({
                    "step_pair": (i, i + 1),
                    "similarity": sim,
                    "step_a": reasoning_steps[i],
                    "step_b": reasoning_steps[i + 1],
                })

        # Goal alignment (if goal provided)
        goal_alignment = None
        goal_sims = []
        if goal:
            goal_emb = self.engine.encode([goal])
            for i, step_emb in enumerate(step_embs):
                sim = cosine_similarity(
                    step_emb.reshape(1, -1), goal_emb
                )[0, 0]
                goal_sims.append(float(sim))
            goal_alignment = float(np.mean(goal_sims))

        # Final score
        if goal_alignment is not None:
            final_score = (
                (1 - self.goal_weight) * seq_coherence
                + self.goal_weight * goal_alignment
            )
        else:
            final_score = seq_coherence

        # Penalty for contradictions
        contradiction_penalty = len(contradictions) * 0.1
        final_score = max(0.0, final_score - contradiction_penalty)

        elapsed = time.perf_counter() - start

        method_parts = [
            f"1. Encoded {len(reasoning_steps)} reasoning steps using "
            f"'{self.engine.model_name}'.",
            f"2. Sequential Coherence: computed cosine similarity between "
            f"each consecutive pair of steps. Mean = {seq_coherence:.4f}.",
            f"3. Contradiction Detection: flagged {len(contradictions)} step pairs "
            f"with similarity < {self.contradiction_threshold} as potential contradictions.",
        ]
        if goal:
            method_parts.append(
                f"4. Goal Alignment: computed cosine similarity of each step to "
                f"the goal. Mean = {goal_alignment:.4f}."
            )
            method_parts.append(
                f"5. Final Score = (1-{self.goal_weight}) * seq_coherence + "
                f"{self.goal_weight} * goal_alignment - "
                f"{len(contradictions)} * 0.1 penalty = {final_score:.4f}."
            )
        else:
            method_parts.append(
                f"4. Final Score = seq_coherence - {len(contradictions)} * 0.1 "
                f"penalty = {final_score:.4f}."
            )

        return MetricResult(
            score=final_score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(final_score, len(contradictions)),
            method="\n".join(method_parts),
            model_used=self.engine.model_name,
            citations=self.citations,
            details={
                "sequential_coherence": seq_coherence,
                "sequential_similarities": sequential_sims,
                "goal_alignment": goal_alignment,
                "goal_similarities": goal_sims if goal else None,
                "contradictions": contradictions,
                "contradiction_penalty": contradiction_penalty,
            },
            elapsed_sec=elapsed,
        )

    def _interpret(self, score: float, n_contradictions: int) -> str:
        parts = []
        if score >= 0.85:
            parts.append("Highly coherent reasoning chain.")
        elif score >= 0.65:
            parts.append("Moderately coherent — some logical gaps.")
        elif score >= 0.4:
            parts.append("Weak coherence — notable logical disconnects.")
        else:
            parts.append("Poor coherence — reasoning chain is fragmented.")
        if n_contradictions > 0:
            parts.append(f" {n_contradictions} potential contradiction(s) detected.")
        return "".join(parts)


class ToolCallAccuracyMetric(BaseMetric):
    """Measures whether an agent selected the correct tools with correct parameters.

    Research Basis:
        Liu, X., Yu, H., Zhang, H., et al. (2023).
        AgentBench: Evaluating LLMs as Agents. ICLR 2024.
        https://arxiv.org/abs/2308.03688

    How Scoring Works:
        Compares predicted tool calls against expected (ground truth) tool calls
        across three dimensions:

        1. **Tool Selection Accuracy**: Did the agent call the right tool?
           Exact string match between predicted and expected tool names.

        2. **Parameter Accuracy**: Did the agent pass the correct arguments?
           For each matched tool call, compare parameter dicts.
           Score = (# matching key-value pairs) / (# expected key-value pairs).

        3. **Ordering Accuracy**: Were tools called in the correct sequence?
           Measured using longest common subsequence (LCS) ratio.

        4. **Final Score** = weighted combination of the three sub-scores.

    Score Range: [0.0, 1.0] where 1.0 = perfect tool usage.

    Parameters:
        selection_weight: Weight for tool selection accuracy (default 0.5).
        param_weight:     Weight for parameter accuracy (default 0.3).
        order_weight:     Weight for ordering accuracy (default 0.2).
    """

    def __init__(
        self,
        selection_weight: float = 0.5,
        param_weight: float = 0.3,
        order_weight: float = 0.2,
    ) -> None:
        total = selection_weight + param_weight + order_weight
        self.selection_weight = selection_weight / total
        self.param_weight = param_weight / total
        self.order_weight = order_weight / total

    @property
    def metric_name(self) -> str:
        return "Tool Call Accuracy"

    @property
    def citations(self) -> List[Citation]:
        return [CITATIONS["agent_eval"]]

    def evaluate(
        self,
        predicted_calls: List[Dict[str, Any]],
        expected_calls: List[Dict[str, Any]],
    ) -> MetricResult:
        """Evaluate tool call accuracy.

        Args:
            predicted_calls: List of dicts with keys:
                - "tool": str (tool name)
                - "params": dict (optional, tool parameters)
            expected_calls: Same format, the ground truth.

        Returns:
            MetricResult with per-call breakdown.
        """
        start = time.perf_counter()

        if not expected_calls:
            elapsed = time.perf_counter() - start
            score = 1.0 if not predicted_calls else 0.0
            return MetricResult(
                score=score,
                metric_name=self.metric_name,
                score_range=(0.0, 1.0),
                interpretation="No expected tool calls." if score == 1.0 else "Agent made unexpected tool calls.",
                method="No expected tool calls to compare against.",
                model_used="N/A (structural comparison)",
                citations=self.citations,
                elapsed_sec=time.perf_counter() - start,
            )

        # 1. Tool Selection Accuracy
        pred_tools = [c.get("tool", "") for c in predicted_calls]
        exp_tools = [c.get("tool", "") for c in expected_calls]

        selection_matches = 0
        for exp_tool in exp_tools:
            if exp_tool in pred_tools:
                selection_matches += 1
        selection_accuracy = selection_matches / len(exp_tools)

        # 2. Parameter Accuracy
        param_scores = []
        per_call = []
        for i, exp_call in enumerate(expected_calls):
            exp_tool = exp_call.get("tool", "")
            exp_params = exp_call.get("params", {})

            # Find matching predicted call
            matched_pred = None
            for pred_call in predicted_calls:
                if pred_call.get("tool", "") == exp_tool:
                    matched_pred = pred_call
                    break

            if matched_pred is None:
                param_scores.append(0.0)
                per_call.append({
                    "expected_tool": exp_tool,
                    "predicted_tool": None,
                    "tool_match": False,
                    "param_score": 0.0,
                })
                continue

            pred_params = matched_pred.get("params", {})
            if not exp_params:
                p_score = 1.0
            else:
                matching_params = sum(
                    1 for k, v in exp_params.items()
                    if pred_params.get(k) == v
                )
                p_score = matching_params / len(exp_params)

            param_scores.append(p_score)
            per_call.append({
                "expected_tool": exp_tool,
                "predicted_tool": matched_pred.get("tool", ""),
                "tool_match": True,
                "param_score": p_score,
                "expected_params": exp_params,
                "predicted_params": pred_params,
            })

        param_accuracy = float(np.mean(param_scores)) if param_scores else 0.0

        # 3. Ordering Accuracy (LCS ratio)
        order_accuracy = self._lcs_ratio(pred_tools, exp_tools)

        # Final score
        final_score = (
            self.selection_weight * selection_accuracy
            + self.param_weight * param_accuracy
            + self.order_weight * order_accuracy
        )

        elapsed = time.perf_counter() - start

        method = (
            f"1. Tool Selection: {selection_matches}/{len(exp_tools)} expected tools "
            f"were called. Accuracy = {selection_accuracy:.4f}.\n"
            f"2. Parameter Accuracy: For each matched tool, compared predicted vs "
            f"expected parameters. Mean accuracy = {param_accuracy:.4f}.\n"
            f"3. Ordering: Longest Common Subsequence ratio between predicted and "
            f"expected tool sequences = {order_accuracy:.4f}.\n"
            f"4. Final = {self.selection_weight:.2f}*selection + "
            f"{self.param_weight:.2f}*params + {self.order_weight:.2f}*order "
            f"= {final_score:.4f}."
        )

        return MetricResult(
            score=final_score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(final_score),
            method=method,
            model_used="N/A (structural comparison)",
            citations=self.citations,
            details={
                "selection_accuracy": selection_accuracy,
                "param_accuracy": param_accuracy,
                "order_accuracy": order_accuracy,
                "per_call": per_call,
                "weights": {
                    "selection": self.selection_weight,
                    "param": self.param_weight,
                    "order": self.order_weight,
                },
            },
            elapsed_sec=elapsed,
        )

    def _lcs_ratio(self, seq1: List[str], seq2: List[str]) -> float:
        """Longest Common Subsequence ratio."""
        if not seq1 or not seq2:
            return 0.0
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]
        return lcs_len / max(m, n)

    def _interpret(self, score: float) -> str:
        if score >= 0.9:
            return "Excellent tool usage — correct tools, parameters, and ordering."
        elif score >= 0.7:
            return "Good tool usage — mostly correct with minor parameter or ordering errors."
        elif score >= 0.5:
            return "Moderate — some tools missed or wrong parameters."
        else:
            return "Poor tool usage — significant errors in tool selection or parameters."


class TaskCompletionMetric(BaseMetric):
    """Graded evaluation of how completely an agent achieved its goal.

    Research Basis:
        Liu, X., et al. (2023). AgentBench. ICLR 2024.
        https://arxiv.org/abs/2308.03688

    How Scoring Works:
        Supports two evaluation modes:

        1. **Checklist mode**: The evaluator provides a list of expected
           sub-goals or criteria. Each is checked against the agent's output
           using embedding similarity.
           Score = (# achieved sub-goals) / (# total sub-goals).

        2. **Reference mode**: Compares the agent's final output against
           a reference output using embedding similarity.
           Score = cosine similarity between output and reference embeddings.

    Score Range: [0.0, 1.0] where 1.0 = all sub-goals achieved / perfect match.

    Parameters:
        engine:    EmbeddingEngine for semantic matching.
        threshold: Similarity threshold for sub-goal achievement (default 0.6).
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        threshold: float = 0.6,
    ) -> None:
        self.engine = engine
        self.threshold = threshold

    @property
    def metric_name(self) -> str:
        return "Task Completion"

    @property
    def citations(self) -> List[Citation]:
        return [CITATIONS["agent_eval"]]

    def evaluate(
        self,
        agent_output: str,
        expected_criteria: Optional[List[str]] = None,
        reference_output: Optional[str] = None,
    ) -> MetricResult:
        """Evaluate task completion.

        Args:
            agent_output:      The agent's final output text.
            expected_criteria: List of sub-goals/criteria to check (checklist mode).
            reference_output:  Reference output to compare against (reference mode).

        At least one of expected_criteria or reference_output must be provided.

        Returns:
            MetricResult with per-criterion or reference comparison details.
        """
        if not expected_criteria and not reference_output:
            raise ValueError(
                "Provide at least one of 'expected_criteria' or 'reference_output'."
            )

        start = time.perf_counter()

        if expected_criteria:
            result = self._evaluate_checklist(agent_output, expected_criteria)
        else:
            result = self._evaluate_reference(agent_output, reference_output)

        result.elapsed_sec = time.perf_counter() - start
        return result

    def _evaluate_checklist(
        self, agent_output: str, criteria: List[str]
    ) -> MetricResult:
        output_emb = self.engine.encode([agent_output])
        criteria_embs = self.engine.encode(criteria)

        sims = cosine_similarity(criteria_embs, output_emb).flatten()
        criterion_results = []
        achieved = 0
        for criterion, sim in zip(criteria, sims):
            met = float(sim) >= self.threshold
            if met:
                achieved += 1
            criterion_results.append({
                "criterion": criterion,
                "similarity": float(sim),
                "threshold": self.threshold,
                "achieved": met,
            })

        score = achieved / len(criteria)

        method = (
            f"1. Encoded agent output and {len(criteria)} expected criteria "
            f"using '{self.engine.model_name}'.\n"
            f"2. Computed cosine similarity between output and each criterion.\n"
            f"3. Criterion achieved if similarity >= {self.threshold}.\n"
            f"4. Score = {achieved}/{len(criteria)} criteria met = {score:.4f}."
        )

        return MetricResult(
            score=score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(score),
            method=method,
            model_used=self.engine.model_name,
            citations=self.citations,
            details={
                "mode": "checklist",
                "criteria_results": criterion_results,
                "achieved": achieved,
                "total": len(criteria),
            },
        )

    def _evaluate_reference(
        self, agent_output: str, reference: str
    ) -> MetricResult:
        output_emb = self.engine.encode([agent_output])
        ref_emb = self.engine.encode([reference])

        sim = float(cosine_similarity(output_emb, ref_emb)[0, 0])
        score = max(0.0, sim)  # clamp negatives

        method = (
            f"1. Encoded agent output and reference output using "
            f"'{self.engine.model_name}'.\n"
            f"2. Computed cosine similarity = {sim:.4f}.\n"
            f"3. Score = max(0, similarity) = {score:.4f}."
        )

        return MetricResult(
            score=score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(score),
            method=method,
            model_used=self.engine.model_name,
            citations=self.citations,
            details={
                "mode": "reference",
                "similarity": sim,
            },
        )

    def _interpret(self, score: float) -> str:
        if score >= 0.9:
            return "Task fully completed — all objectives met."
        elif score >= 0.7:
            return "Mostly completed — most objectives achieved."
        elif score >= 0.5:
            return "Partially completed — some objectives met."
        elif score >= 0.2:
            return "Minimally completed — few objectives achieved."
        else:
            return "Task not completed."
