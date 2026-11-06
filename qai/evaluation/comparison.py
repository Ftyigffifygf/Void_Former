"""Comparison utilities between classical and quantum architectures."""

from __future__ import annotations
from typing import Dict, Any, List


class ModelComparator:
    """Aggregates and formats benchmark comparisons."""

    @staticmethod
    def compare_models(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarizes learning, quantum, and computational metrics across evaluated models."""
        summary = {
            "models_evaluated": len(results),
            "results": results,
        }
        return summary
