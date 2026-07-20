"""Simple data preparation and one-call tabular machine learning."""

from .api import TrainingResult, classify, regress
from .data import DatasetSplit, load_table, prepare_dataset

__all__ = [
    "DatasetSplit",
    "TrainingResult",
    "classify",
    "load_table",
    "prepare_dataset",
    "regress",
]
