"""Data loading, validation, preprocessing, and train/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TableLike = Union[str, Path, pd.DataFrame]
Task = Literal["classification", "regression"]


@dataclass
class DatasetSplit:
    """Validated data and its reusable preprocessing pipeline.

    ``random_state=None`` intentionally leaves the train/test split random.
    Pass an integer to reproduce a split exactly.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    target: str
    task: Task


def load_table(data: TableLike) -> pd.DataFrame:
    """Load a DataFrame or a CSV, TSV, or Parquet file into a DataFrame."""

    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Supported file types are .csv, .tsv, .parquet, and .pq")


def prepare_dataset(
    data: TableLike,
    *,
    target: str,
    onehot: Iterable[str] | None = None,
    task: Task,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> DatasetSplit:
    """Prepare a tabular dataset for a classifier or regressor.

    Columns listed in ``onehot`` are imputed and one-hot encoded. Every other
    feature is passed through as numeric data after median imputation, keeping
    the public API explicit and predictable.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    frame = load_table(data)
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} was not found")
    if frame.empty:
        raise ValueError("The dataset is empty")

    onehot_columns = list(dict.fromkeys(onehot or []))
    unknown = sorted(set(onehot_columns) - set(frame.columns))
    if unknown:
        raise ValueError(f"onehot columns were not found: {unknown}")
    if target in onehot_columns:
        raise ValueError("target cannot also be a onehot column")

    y = frame.pop(target)
    if y.isna().any():
        raise ValueError("target contains missing values; remove or fill them first")
    X = frame
    if X.shape[1] == 0:
        raise ValueError("The dataset needs at least one feature column")

    numeric_columns = [column for column in X.columns if column not in onehot_columns]
    non_numeric = [column for column in numeric_columns if not pd.api.types.is_numeric_dtype(X[column])]
    if non_numeric:
        raise ValueError(
            "Non-numeric feature columns must be listed in onehot: "
            f"{non_numeric}"
        )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_columns))
    if onehot_columns:
        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("onehot", categorical, onehot_columns))

    preprocessor = ColumnTransformer(transformers=transformers)
    stratify = _stratify_target(y, task)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        target=target,
        task=task,
    )


def _stratify_target(y: pd.Series, task: Task) -> pd.Series | None:
    """Stratify classifications only when every class can support a split."""

    if task != "classification":
        return None
    counts = y.value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return y
