# too-simple-ai

一键式表格机器学习，提供稳妥的默认配置。输入一个 CSV 文件（或
DataFrame），指定目标列，并可选地列出类别特征列，即可获得调优后的模型和
清晰的评估指标。

## 安装

```bash
pip install too-simple-ai
```

如需使用可选的 XGBoost 和 LightGBM 模型：

```bash
pip install 'too-simple-ai[boost]'
```

## 快速开始

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

标签预测请使用 `classify(...)`；数值目标预测请使用 `regress(...)`。输入可以是
CSV/TSV/Parquet 文件路径，也可以是 pandas DataFrame。

## 搜索规模

搜索规模采用熟悉的 YOLO 风格。以下是在 Iris 等小型数据集上的目标预算；大型
数据集会在调优时自动采样。

| 搜索规模 | 目标预算 | CV 折数 | 每个模型的候选配置 | 调优最大行数 |
| --- | ---: | ---: | ---: | ---: |
| `n` / `nano` | ~3 秒 | 2 | 1 | 5,000 |
| `s` / `small` | ~10 秒 | 3 | 2 | 10,000 |
| `m` / `medium` | ~30 秒 | 3 | 4 | 30,000 |
| `l` / `large` | ~60 秒 | 4 | 8 | 75,000 |
| `x` / `xlarge` | ~120 秒 | 5 | 16 | 150,000 |

时间为估计值，并非硬性超时限制；实际时长受硬件、特征和所选模型影响。可传入
`models=["random_forest", "svm"]` 来限制本次运行的模型。

## 自动完成的处理

- 数值列会以中位数填补缺失值，并直接用于训练。
- `onehot` 中指定的列会以众数填补缺失值，并进行独热编码。
- 分类任务会在可行时采用分层训练集/测试集划分。
- 固定的 `random_state` 可使数据划分和模型训练可复现；省略它则每次都会生成新的随机划分。
- 分类任务仅在每个交叉验证折内通过随机过采样平衡训练数据；留出的测试集绝不会被重采样。
- 常见的 sklearn 模型会自动排序：逻辑回归、随机森林、极端随机树、SVM 和朴素贝叶斯；安装可选依赖后还会加入 XGBoost 和 LightGBM。

分类任务的 `result.leaderboard` 会报告 `accuracy`、`balanced_accuracy`、
`precision`、`recall`、`f1` 和 `roc_auc`；回归任务则报告 `mae`、`rmse` 和 `r2`。

## 依赖项

`too-simple-ai` 当前依赖：

- [PyTorch](https://pytorch.org/)（`torch`）
- [scikit-learn](https://scikit-learn.org/)（`scikit-learn`）
- [pandas](https://pandas.pydata.org/)（`pandas`）
- [imbalanced-learn](https://imbalanced-learn.org/)（`imbalanced-learn`）

## 状态

`0.2.0` 版本加入了首个公开的表格机器学习 API。

## 许可证

MIT。参见 [LICENSE](LICENSE)。
