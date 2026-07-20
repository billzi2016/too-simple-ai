# too-simple-ai

One-call tabular machine learning with safe defaults. Give it a CSV (or a
DataFrame), name the target column, optionally list categorical columns, and
get tuned models and clear metrics back.

## Installation

```bash
pip install too-simple-ai
```

For optional XGBoost and LightGBM models:

```bash
pip install 'too-simple-ai[boost]'
```

## Quick start

```python
from too_simple_ai.ml import classify

result = classify(
    "customers.csv",
    target="churned",
    onehot=["city", "plan"],
    search="s",
    random_state=42,
)

print(result.leaderboard)
predictions = result.predict("new_customers.csv")
```

Use `classify(...)` for labels or `regress(...)` for numeric targets. Input can
be a CSV/TSV/Parquet path or a pandas DataFrame.

## Search sizes

Search size follows the familiar YOLO-style scale. These are target budgets on
small data such as Iris; large datasets are sampled for tuning automatically.

| Search | Target budget | CV folds | Candidate settings/model | Max rows for tuning |
| --- | ---: | ---: | ---: | ---: |
| `n` / `nano` | ~3 s | 2 | 1 | 5,000 |
| `s` / `small` | ~10 s | 3 | 2 | 10,000 |
| `m` / `medium` | ~30 s | 3 | 4 | 30,000 |
| `l` / `large` | ~60 s | 4 | 8 | 75,000 |
| `x` / `xlarge` | ~120 s | 5 | 16 | 150,000 |

Time is an estimate, not a hard timeout. It varies with hardware, features,
and selected models. Pass `models=["random_forest", "svm"]` to limit a run.

## What happens automatically

- Numeric columns are median-imputed and used directly.
- Columns named in `onehot` are mode-imputed and one-hot encoded.
- Classification uses a stratified train/test split when possible.
- A fixed `random_state` makes the split and model training reproducible; omit
  it for a new random split on every run.
- Classification balances its training data with random oversampling inside
  each CV fold only. The held-out test set is never resampled.
- Common sklearn models are ranked automatically: logistic regression, random
  forest, extra trees, SVM, and naive Bayes; XGBoost and LightGBM join when
  their optional dependency is installed.

`result.leaderboard` reports classification `accuracy`, `balanced_accuracy`,
`precision`, `recall`, `f1`, and `roc_auc`; regression reports `mae`, `rmse`,
and `r2`.

## Dependencies

`too-simple-ai` currently depends on:

- [PyTorch](https://pytorch.org/) (`torch`)
- [scikit-learn](https://scikit-learn.org/) (`scikit-learn`)
- [pandas](https://pandas.pydata.org/) (`pandas`)
- [imbalanced-learn](https://imbalanced-learn.org/) (`imbalanced-learn`)

## Status

Version `0.2.0` adds the first public tabular ML API.

## License

MIT. See [LICENSE](LICENSE).
