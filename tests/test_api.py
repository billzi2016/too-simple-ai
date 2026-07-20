import unittest

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris

from too_simple_ai.ml import classify, regress


class ApiTests(unittest.TestCase):
    def test_classify_returns_metrics_and_predictions(self):
        iris = load_iris(as_frame=True)
        frame = iris.frame.rename(columns={"target": "species"})
        result = classify(
            frame,
            target="species",
            search="n",
            random_state=42,
            models=["logistic_regression"],
        )

        row = result.leaderboard.iloc[0]
        self.assertEqual(row["model"], "logistic_regression")
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"):
            self.assertIn(metric, row.index)
        self.assertEqual(len(result.predict(frame.drop(columns=["species"]).head())), 5)

    def test_regress_returns_regression_metrics(self):
        diabetes = load_diabetes(as_frame=True)
        frame = diabetes.frame
        result = regress(
            frame,
            target="target",
            search="n",
            random_state=42,
            models=["linear_regression"],
        )

        row = result.leaderboard.iloc[0]
        self.assertEqual(row["model"], "linear_regression")
        for metric in ("mae", "rmse", "r2"):
            self.assertIn(metric, row.index)


if __name__ == "__main__":
    unittest.main()
