import unittest

import pandas as pd

from too_simple_ai.ml.data import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "age": list(range(20, 40)),
                "city": ["A", "B"] * 10,
                "label": [0, 1] * 10,
            }
        )

    def test_fixed_seed_reproduces_split_and_onehot_preprocessor(self):
        first = prepare_dataset(
            self.frame,
            target="label",
            onehot=["city"],
            task="classification",
            random_state=42,
        )
        second = prepare_dataset(
            self.frame,
            target="label",
            onehot=["city"],
            task="classification",
            random_state=42,
        )

        self.assertListEqual(first.X_train.index.tolist(), second.X_train.index.tolist())
        self.assertEqual(first.preprocessor.fit_transform(first.X_train).shape[1], 3)

    def test_non_numeric_feature_requires_onehot(self):
        with self.assertRaisesRegex(ValueError, "must be listed in onehot"):
            prepare_dataset(self.frame, target="label", task="classification")

    def test_onehot_requires_an_iterable_and_task_must_be_known(self):
        with self.assertRaisesRegex(TypeError, "not a single string"):
            prepare_dataset(self.frame, target="label", onehot="city", task="classification")
        with self.assertRaisesRegex(ValueError, "task must be"):
            prepare_dataset(self.frame, target="label", onehot=["city"], task="clustering")


if __name__ == "__main__":
    unittest.main()
