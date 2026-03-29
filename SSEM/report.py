"""
Evaluation Report
=================

Aggregates multiple SSEM metric results into a single, structured
evaluation report with full transparency.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from SSEM.core import MetricResult


class EvaluationReport:
    """Aggregated evaluation report from multiple SSEM metrics.

    Provides:
    - Summary table of all scores
    - Full transparency report for each metric
    - JSON/dict export for downstream tooling
    - Citation bibliography

    Usage:
        report = EvaluationReport()
        report.add(bertscore_result)
        report.add(faithfulness_result)
        print(report.summary())
        print(report.explain())
    """

    def __init__(self) -> None:
        self.results: List[MetricResult] = []

    def add(self, result: MetricResult) -> None:
        """Add a metric result to the report."""
        self.results.append(result)

    def summary(self) -> str:
        """One-line-per-metric summary table."""
        if not self.results:
            return "No metrics evaluated yet."

        lines = ["=" * 70, "SSEM Evaluation Report", "=" * 70, ""]
        max_name = max(len(r.metric_name) for r in self.results)
        header = f"{'Metric':<{max_name+2}} {'Score':>8}  {'Range':>12}  Interpretation"
        lines.append(header)
        lines.append("-" * len(header))
        for r in self.results:
            rng = f"[{r.score_range[0]}, {r.score_range[1]}]"
            lines.append(
                f"{r.metric_name:<{max_name+2}} {r.score:>8.4f}  {rng:>12}  {r.interpretation}"
            )
        lines.append("")
        total_time = sum(r.elapsed_sec or 0 for r in self.results)
        lines.append(f"Total evaluation time: {total_time:.3f}s")
        lines.append("=" * 70)
        return "\n".join(lines)

    def explain(self) -> str:
        """Full transparency report with method and citations for each metric."""
        sections = [self.summary(), ""]
        for r in self.results:
            sections.append(r.explain())
            sections.append("")
        sections.append(self._bibliography())
        return "\n".join(sections)

    def _bibliography(self) -> str:
        """Deduplicated bibliography of all cited papers."""
        seen = set()
        citations = []
        for r in self.results:
            for c in r.citations:
                key = c.url
                if key not in seen:
                    seen.add(key)
                    citations.append(c)
        lines = ["=" * 70, "Bibliography", "=" * 70]
        for i, c in enumerate(citations, 1):
            lines.append(f"[{i}] {c}")
            lines.append(f"    {c.description}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export report as a dict (JSON-serializable)."""
        return {
            "metrics": {r.metric_name: r.to_dict() for r in self.results},
            "total_time": sum(r.elapsed_sec or 0 for r in self.results),
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def scores(self) -> Dict[str, float]:
        """Simple dict of metric_name -> score."""
        return {r.metric_name: r.score for r in self.results}
