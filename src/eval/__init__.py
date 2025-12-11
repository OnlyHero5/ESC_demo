"""评测模块"""

from .metrics import (
    compute_bleu,
    compute_rouge,
    compute_distinct,
    compute_all_metrics,
    compute_bertscore
)

__all__ = [
    "compute_bleu",
    "compute_rouge",
    "compute_distinct",
    "compute_all_metrics",
    "compute_bertscore"
]