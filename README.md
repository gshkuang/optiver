# Optiver 项目说明

## 概览
- 目标：基于 Kaggle Optiver Trading at the Close 任务，整理与注释两套公开基线方案，便于复现与学习。
- 竞赛主页：`https://www.kaggle.com/competitions/optiver-trading-at-the-close`
- 飞书文档：`https://dqpzque4vfe.feishu.cn/wiki/OlNPwx4uui6OflkB0D3c2iZOn2d`

## 文件说明
- `baseline-lgb-xgb-and-catboost.ipynb`：三模型基线（LGB/XGB/CatBoost），含基础特征工程与竞赛环境推断。
- `optiver-robust-best-single-model.ipynb`：Robust Best 单模型（LightGBM），含分组/时序特征工程、离线时序划分与在线推断。

## 代码出处
- `baseline-lgb-xgb-and-catboost.ipynb` 来源：`https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost`
- `optiver-robust-best-single-model.ipynb` 来源：`https://www.kaggle.com/code/lblhandsome/optiver-robust-best-single-model`

## 依据飞书文档的结构化说明
- Optiver 介绍：作为全球领先做市商，强调以技术驱动的量化交易与流动性供给；竞赛聚焦收盘阶段（Closing Auction）的价格/方向预测。
- 两个方案对应关系：
  - 基线（LGB/XGB/CatBoost）：强调稳健与可复现，便于快速建立参考与做误差分析。
  - 进阶（LightGBM 单模型）：通过更丰富的分组与时序特征提升表达能力，并严格使用时间顺序进行训练/验证以避免泄露。
- 后续建议：在上述基线之上进一步探索 GBDT/Transformer、更细粒度时间窗特征与交易规则约束，以更贴近真实交易决策流程。

## 使用指南
- 准备数据：从 Kaggle 获取比赛数据，按原 Notebook 的路径使用（Kaggle 环境以 `/kaggle/input/optiver-trading-at-the-close/` 为根）。本地使用时可自行调整数据路径。
- 依赖：`python3`、`pandas`、`numpy`、`scikit-learn`，并根据 Notebook 安装 `lightgbm`、`xgboost`、`catboost`。
- 运行：在对应 Notebook 中依次执行各单元。若使用比赛的交互环境（`optiver2023`），需在 Kaggle 上运行以加载官方评测接口。

## 注意事项
- Notebook 已加入解释性 Markdown 注释，未改变核心实现逻辑。若需在本地运行，请根据实际环境调整数据路径与依赖。
- 竞赛推断接口（`optiver2023`）为 Kaggle 专用，在本地环境不可直接使用。

## 参考
- Kaggle 比赛页：`https://www.kaggle.com/competitions/optiver-trading-at-the-close`
- 飞书文档（项目背景与方案）：`https://dqpzque4vfe.feishu.cn/wiki/OlNPwx4uui6OflkB0D3c2iZOn2d`
