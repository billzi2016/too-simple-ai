"""Model definitions and bounded search grids for simple tabular ML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC, SVR

Task = Literal["classification", "regression"]


def to_dense(matrix):
    """Convert sparse preprocessing output for estimators such as GaussianNB."""

    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict[str, list]


def model_specs(task: Task, random_state: int | None) -> list[ModelSpec]:
    """Return dependable sklearn models plus installed optional boosters."""

    if task == "classification":
        specs = [
            ModelSpec(
                "logistic_regression",
                LogisticRegression(max_iter=1_000),
                {"model__C": [0.1, 1.0, 10.0]},
            ),
            ModelSpec(
                "random_forest",
                RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
                {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 2]},
            ),
            ModelSpec(
                "extra_trees",
                ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
                {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 2]},
            ),
            ModelSpec(
                "svm",
                SVC(probability=True, random_state=random_state),
                {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
            ),
            ModelSpec(
                "naive_bayes",
                SklearnPipeline(
                    [("densify", FunctionTransformer(to_dense)), ("model", GaussianNB())]
                ),
                {"model__model__var_smoothing": [1e-11, 1e-9, 1e-7]},
            ),
        ]
        return specs + _optional_classifiers(random_state)

    specs = [
        ModelSpec("linear_regression", LinearRegression(n_jobs=-1), {}),
        ModelSpec(
            "random_forest",
            RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state),
            {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 2]},
        ),
        ModelSpec(
            "extra_trees",
            ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=random_state),
            {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 2]},
        ),
        ModelSpec(
            "svr",
            SVR(),
            {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.01, 0.1, 0.5]},
        ),
    ]
    return specs + _optional_regressors(random_state)


def _optional_classifiers(random_state: int | None) -> list[ModelSpec]:
    specs = []
    try:
        from xgboost import XGBClassifier

        specs.append(
            ModelSpec(
                "xgboost",
                XGBClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
                {"model__max_depth": [3, 6, 10], "model__learning_rate": [0.03, 0.1]},
            )
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        specs.append(
            ModelSpec(
                "lightgbm",
                LGBMClassifier(n_estimators=200, n_jobs=-1, random_state=random_state, verbosity=-1),
                {"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.03, 0.1]},
            )
        )
    except ImportError:
        pass
    return specs


def _optional_regressors(random_state: int | None) -> list[ModelSpec]:
    specs = []
    try:
        from xgboost import XGBRegressor

        specs.append(
            ModelSpec(
                "xgboost",
                XGBRegressor(n_estimators=200, n_jobs=-1, random_state=random_state),
                {"model__max_depth": [3, 6, 10], "model__learning_rate": [0.03, 0.1]},
            )
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        specs.append(
            ModelSpec(
                "lightgbm",
                LGBMRegressor(n_estimators=200, n_jobs=-1, random_state=random_state, verbosity=-1),
                {"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.03, 0.1]},
            )
        )
    except ImportError:
        pass
    return specs
