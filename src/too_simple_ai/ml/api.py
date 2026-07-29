"""One-function interfaces for tabular classification and regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, KFold, train_test_split

from .data import DatasetSplit, TableLike, load_table, prepare_dataset
from .models import ModelSpec, model_specs

Task = Literal["classification", "regression"]

_SIZE_ALIASES = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}
_SEARCH_PROFILES = {
    "n": {"cv": 2, "candidates": 1, "max_rows": 5_000},
    "s": {"cv": 3, "candidates": 2, "max_rows": 10_000},
    "m": {"cv": 3, "candidates": 4, "max_rows": 30_000},
    "l": {"cv": 4, "candidates": 8, "max_rows": 75_000},
    "x": {"cv": 5, "candidates": 16, "max_rows": 150_000},
}


@dataclass
class ModelResult:
    """One fitted model and its held-out test metrics."""

    name: str
    estimator: object
    metrics: dict[str, float | None]
    best_params: dict[str, object]


@dataclass
class TrainingResult:
    """A fitted model collection with a simple leaderboard and prediction API."""

    task: Task
    target: str
    models: dict[str, ModelResult]

    @property
    def leaderboard(self) -> pd.DataFrame:
        """Return all held-out metrics, sorted by the primary task metric."""

        rows = [
            {"model": result.name, **result.metrics, "best_params": result.best_params}
            for result in self.models.values()
        ]
        primary = "f1" if self.task == "classification" else "r2"
        return pd.DataFrame(rows).sort_values(primary, ascending=False, na_position="last").reset_index(drop=True)

    @property
    def best_model(self) -> ModelResult:
        """Return the top held-out model."""

        best_name = self.leaderboard.iloc[0]["model"]
        return self.models[best_name]

    def predict(self, data: TableLike):
        """Predict with the best model from a CSV/DataFrame of feature rows."""

        frame = load_table(data)
        return self.best_model.estimator.predict(frame.drop(columns=[self.target], errors="ignore"))


def classify(
    data: TableLike,
    *,
    target: str,
    onehot: Iterable[str] | None = None,
    search: str = "s",
    random_state: int | None = None,
    test_size: float = 0.2,
    balance: bool = True,
    models: Iterable[str] | None = None,
) -> TrainingResult:
    """Train, tune, and rank common classification models in one call.

    ``search`` accepts ``n/s/m/l/x`` or their text aliases. Balancing uses
    random oversampling inside every CV training fold only.
    """

    return _train(
        data,
        target=target,
        onehot=onehot,
        task="classification",
        search=search,
        random_state=random_state,
        test_size=test_size,
        balance=balance,
        models=models,
    )


def regress(
    data: TableLike,
    *,
    target: str,
    onehot: Iterable[str] | None = None,
    search: str = "s",
    random_state: int | None = None,
    test_size: float = 0.2,
    models: Iterable[str] | None = None,
) -> TrainingResult:
    """Train, tune, and rank common regression models in one call."""

    return _train(
        data,
        target=target,
        onehot=onehot,
        task="regression",
        search=search,
        random_state=random_state,
        test_size=test_size,
        balance=False,
        models=models,
    )


def _train(
    data: TableLike,
    *,
    target: str,
    onehot: Iterable[str] | None,
    task: Task,
    search: str,
    random_state: int | None,
    test_size: float,
    balance: bool,
    models: Iterable[str] | None,
) -> TrainingResult:
    profile_name = _normalise_profile(search)
    split = prepare_dataset(
        data,
        target=target,
        onehot=onehot,
        task=task,
        test_size=test_size,
        random_state=random_state,
    )
    selected = _select_models(model_specs(task, random_state), models)
    profile = _SEARCH_PROFILES[profile_name]
    results = {
        spec.name: _fit_model(spec, split, task, profile, random_state, balance)
        for spec in selected
    }
    return TrainingResult(task=task, target=target, models=results)


def _fit_model(
    spec: ModelSpec,
    split: DatasetSplit,
    task: Task,
    profile: dict[str, int],
    random_state: int | None,
    balance: bool,
) -> ModelResult:
    sampler = RandomOverSampler(random_state=random_state) if balance else "passthrough"
    pipeline = Pipeline(
        [("preprocessor", clone(split.preprocessor)), ("balance", sampler), ("model", spec.estimator)]
    )
    X_search, y_search = _search_sample(split, task, profile["max_rows"], random_state)
    cv = _cross_validator(y_search, task, profile["cv"], random_state)
    candidates = list(ParameterGrid(spec.param_grid)) or [{}]
    limited = candidates[: profile["candidates"]]
    search = GridSearchCV(
        pipeline,
        param_grid=[{key: [value] for key, value in params.items()} for params in limited],
        scoring="f1_weighted" if task == "classification" else "r2",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_search, y_search)
    estimator = clone(search.best_estimator_).fit(split.X_train, split.y_train)
    predictions = estimator.predict(split.X_test)
    metrics = _classification_metrics(split.y_test, predictions, estimator, split.X_test) if task == "classification" else _regression_metrics(split.y_test, predictions)
    return ModelResult(spec.name, estimator, metrics, search.best_params_)


def _search_sample(
    split: DatasetSplit, task: Task, max_rows: int, random_state: int | None
) -> tuple[pd.DataFrame, pd.Series]:
    if len(split.X_train) <= max_rows:
        return split.X_train, split.y_train
    stratify = split.y_train if task == "classification" else None
    X_search, _, y_search, _ = train_test_split(
        split.X_train,
        split.y_train,
        train_size=max_rows,
        random_state=random_state,
        stratify=stratify,
    )
    return X_search, y_search


def _cross_validator(y: pd.Series, task: Task, requested_folds: int, random_state: int | None):
    if task == "classification":
        folds = min(requested_folds, int(y.value_counts().min()))
        if folds < 2:
            raise ValueError("Each class needs at least two training samples for GridSearchCV")
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    folds = min(requested_folds, len(y))
    if folds < 2:
        raise ValueError("At least two training samples are required for GridSearchCV")
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def _classification_metrics(y_true, predictions, estimator, X_test) -> dict[str, float | None]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "roc_auc": None,
    }
    if not hasattr(estimator, "predict_proba"):
        return metrics
    try:
        probabilities = estimator.predict_proba(X_test)
        if probabilities.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted")
            )
    except ValueError:
        pass
    return metrics


def _regression_metrics(y_true, predictions) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _normalise_profile(search: str) -> str:
    if not isinstance(search, str):
        raise TypeError("search must be a string such as 's' or 'small'")
    profile = _SIZE_ALIASES.get(search.lower(), search.lower())
    if profile not in _SEARCH_PROFILES:
        choices = ", ".join(_SEARCH_PROFILES)
        raise ValueError(f"search must be one of: {choices}")
    return profile


def _select_models(specs: list[ModelSpec], requested: Iterable[str] | None) -> list[ModelSpec]:
    if requested is None:
        return specs
    if isinstance(requested, str):
        raise TypeError("models must be an iterable of model names, not a single string")
    requested_names = list(requested)
    if not requested_names:
        raise ValueError("models must contain at least one model name")
    available = {spec.name: spec for spec in specs}
    unknown = sorted(set(requested_names) - set(available))
    if unknown:
        raise ValueError(f"Unknown or unavailable models: {unknown}")
    return [available[name] for name in requested_names]
