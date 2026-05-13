# DAS 鸟类音节训练：流程与概念说明

本文说明 `deepLearning/train_core.py` 里的训练在做什么，以及与完整 `das/train.py` 的差别。

## 整体流程（端到端）

1. **数据**（`prepare` 生成）：一段波形 \(x[t]\) 与对齐的标签矩阵 \(y[t,c]\)。每一时刻 \(t\) 对应一行「类别概率」（含静音类 `noise`），由手工 `_annotations.csv` 转成帧级 one-hot / 归一化概率（见 `das.make_dataset.normalize_probabilities`）。
2. **滑窗**：网络不看单个采样点，而看长度为 **`nb_hist`** 的一段上下文（默认 1024 点）。对每个窗口，网络输出 **当前中心时刻**（或对齐时刻）的类别分布——这叫 **帧级分类（frame-wise classification）**。
3. **前端 + TCN**：默认模型名 `tcn` 在 DAS 里对应 **`tcn_stft`**：可先 STFT 把波形变成时频表示，再堆叠 **时序卷积网络（TCN）**——膨胀卷积（dilations 如 1,2,4,8,16）用大感受野捕获音节结构，参数量相对 LSTM 更易并行。
4. **损失**：多类时用 **`categorical_crossentropy`**（在 `das.models` 里编译进模型）：预测概率与 \(y\) 对齐。
5. **验证与早停**：用验证集 `val_loss`，`ModelCheckpoint` 存最优权重，`EarlyStopping` 防止过拟合。

推理阶段（`predict`）：由 **`deepLearning/predict_core.py`** 调度（调用 `das.predict.predict` 做前向与分段，不写死在 `cli_predict`）；输出与 DAS CLI 一致，仍依赖同一套 `_params.yaml` / `_model.h5`。

## 核心概念（为什么要帧级 + TCN）

| 概念 | 含义 |
|------|------|
| **帧级标签** | 每个时间点属于「噪声」还是某个音节类型；比「整段录音一个标签」更细，适合做边界检测。 |
| **`nb_hist`** | 模型每次决策看到的采样点数；越大上下文越长，显存与计算也越大。 |
| **`stride=1`（分类默认）** | 相邻窗口在时间轴上每次移动 1 个采样（配合 center label），密集预测。 |
| **`y_offset ≈ nb_hist/2`** | 标签取窗口中心时刻，使决策对齐声学事件的中间而非边界。 |
| **TCN / dilated conv** | 用多层空洞卷积在同一帧上看到更远的前后文，适合鸟鸣这类短时结构化声音。 |

## `train_core.py` 与完整 `das.train.train` 的差异

本仓库的 **`run_classification_training`** **只保留「分类 + 标准拟合」主干**：

- **保留**：`io.load`、`AudioSequence`、`models.model_dict`、`utils.save_params`、checkpoint / early stopping（与上游一致，便于 `predict`）。
- **省略**：数据增强、wandb/tensorboard、`post_opt` 网格搜索后处理调参、完整 test 集评测报表等（减少分支与依赖；需要时可回到上游 `das.train.train`）。

因此：**不是从零重写网络数学**，而是把「训练循环」收到本地，结构更清晰；**仍依赖已安装的 `das` 包**（模型定义、数据管线与推理格式不变）。

## 相关文件

- `deepLearning/dataset.py`：WAV + `*_annotations.csv` → `*.npy` 数据集。
- `deepLearning/train_core.py`：精简训练入口。
- `das/docs/technical/data_formats.md`：官方数据字典约定。
