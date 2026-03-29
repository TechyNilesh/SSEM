"""
Code Evaluation Metrics
=======================

Execution-based evaluation of generated code.

1. **CodeCorrectnessMetric** — Pass@k metric: run generated code against
   test cases and measure functional correctness.

Research Basis:
    Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code.
    https://arxiv.org/abs/2107.03374

Security Note:
    Code is executed in a subprocess with a timeout.  For production use,
    consider sandboxing (Docker, gVisor) to isolate untrusted code execution.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np

from SSEM.core import (
    BaseMetric,
    Citation,
    CITATIONS,
    MetricResult,
)


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k from Chen et al. (2021).

    Args:
        n: Total number of samples generated.
        c: Number of samples that passed all tests.
        k: k in pass@k.

    Returns:
        Estimated probability that at least one of k random samples passes.

    Formula:
        pass@k = 1 - C(n-c, k) / C(n, k)

    This is the unbiased estimator from Equation 1 of the Codex paper.
    It avoids the biased estimate of simply computing (c/n)^k.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c - k + 1, n - c + 1)) / math.prod(range(n - k + 1, n + 1))


class CodeCorrectnessMetric(BaseMetric):
    """Execution-based code correctness evaluation using Pass@k.

    Research Basis:
        Chen, M., Tworek, J., Jun, H., et al. (2021).
        Evaluating Large Language Models Trained on Code.
        https://arxiv.org/abs/2107.03374

    How Scoring Works:
        1. For each code sample, write it to a temporary file.
        2. Append the provided test cases (assertions).
        3. Execute in a subprocess with a timeout.
        4. A sample **passes** if execution exits with code 0 (all assertions hold).
        5. Compute **Pass@k** using the unbiased estimator:
           pass@k = 1 - C(n-c, k) / C(n, k)
           where n = total samples, c = passing samples, k = desired k.

    Score Range: [0.0, 1.0] where 1.0 = all samples pass.

    Parameters:
        timeout: Max execution time per sample in seconds. Default 10.
        k_values: List of k values to compute pass@k for. Default [1, 5, 10].
    """

    def __init__(
        self,
        timeout: int = 10,
        k_values: Optional[List[int]] = None,
    ) -> None:
        self.timeout = timeout
        self.k_values = k_values or [1]

    @property
    def metric_name(self) -> str:
        return "Code Correctness (Pass@k)"

    @property
    def citations(self) -> List[Citation]:
        return [CITATIONS["pass_at_k"]]

    def evaluate(
        self,
        code_samples: List[str],
        test_code: str,
    ) -> MetricResult:
        """Evaluate code correctness.

        Args:
            code_samples: List of code strings (multiple attempts at the same problem).
            test_code:    Test code (assertions) to append after each sample.

        Returns:
            MetricResult with pass@k scores and per-sample execution details.
        """
        start = time.perf_counter()

        n = len(code_samples)
        sample_results = []
        pass_count = 0

        for i, code in enumerate(code_samples):
            passed, error = self._execute(code, test_code)
            if passed:
                pass_count += 1
            sample_results.append({
                "sample_index": i,
                "passed": passed,
                "error": error,
            })

        # Compute pass@k for each k
        pass_at_k_scores = {}
        for k in self.k_values:
            if k > n:
                pass_at_k_scores[f"pass@{k}"] = None  # not enough samples
            else:
                pass_at_k_scores[f"pass@{k}"] = _pass_at_k(n, pass_count, k)

        # Primary score = pass@1 (or first available k)
        primary_k = self.k_values[0]
        primary_score = pass_at_k_scores.get(f"pass@{primary_k}", 0.0) or 0.0

        elapsed = time.perf_counter() - start

        method = (
            f"1. Executed {n} code samples against the provided test cases.\n"
            f"2. Each sample was written to a temp file, test code appended, "
            f"and run as a subprocess with {self.timeout}s timeout.\n"
            f"3. A sample passes if exit code = 0 (all assertions hold).\n"
            f"4. Results: {pass_count}/{n} samples passed.\n"
            f"5. Pass@k computed using the unbiased estimator from Chen et al. (2021):\n"
            f"   pass@k = 1 - C(n-c, k) / C(n, k)\n"
            f"   where n={n}, c={pass_count}.\n"
            f"6. Scores: {pass_at_k_scores}"
        )

        return MetricResult(
            score=primary_score,
            metric_name=self.metric_name,
            score_range=(0.0, 1.0),
            interpretation=self._interpret(primary_score, pass_count, n),
            method=method,
            model_used="N/A (execution-based)",
            citations=self.citations,
            details={
                "pass_count": pass_count,
                "total_samples": n,
                "pass_at_k": pass_at_k_scores,
                "sample_results": sample_results,
                "timeout": self.timeout,
            },
            elapsed_sec=elapsed,
        )

    def _execute(self, code: str, test_code: str) -> tuple:
        """Execute code + tests in a subprocess.

        Returns:
            (passed: bool, error: Optional[str])
        """
        full_code = code + "\n" + test_code
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(full_code)
                f.flush()
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr[:500] if result.stderr else "Non-zero exit code"
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out ({self.timeout}s)"
        except Exception as e:
            return False, str(e)[:500]

    def _interpret(self, score: float, passed: int, total: int) -> str:
        if score >= 0.95:
            return f"Excellent — {passed}/{total} samples pass. Code is functionally correct."
        elif score >= 0.7:
            return f"Good — {passed}/{total} samples pass. Minor issues in some attempts."
        elif score >= 0.3:
            return f"Partial — {passed}/{total} samples pass. Significant correctness issues."
        else:
            return f"Poor — {passed}/{total} samples pass. Code is largely incorrect."
