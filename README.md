# Optiver 项目说明

## 概览
- 目标：基于 Kaggle Optiver Trading at the Close 任务，构建两套可运行的基线方案以预测收盘阶段的价格或方向变化。
- 竞赛主页：`https://www.kaggle.com/competitions/optiver-trading-at-the-close`
- 本仓库提供两份独立 Notebook，分别实现通用基线与分组时序增强方案。

## 文件结构
- `Optiver_Scheme1_Baseline.ipynb`：方案一，数值标准化 + 类别独热编码 + Ridge/LogisticRegression
- `Optiver_Scheme2_GroupTime_RF.ipynb`：方案二，分组与时序特征工程 + RandomForest
- `Optiver_Trading_At_Close_Notebook.ipynb`：合并版（保留以供参考）
- `submission_scheme1.csv` / `submission_scheme2.csv`：推断输出（运行 Notebook 后生成，若存在 `test.csv`）

## 数据与依赖
- 数据文件：将 `train.csv` 与（可选）`test.csv` 放置到项目根目录或 `./data/` 目录下
- 依赖：`python3`、`pandas`、`numpy`、`scikit-learn`

## 快速开始
1. 准备数据：将 `train.csv` 和（可选）`test.csv` 放入根目录或 `./data/`
2. 运行方案一：打开并依次执行 `Optiver_Scheme1_Baseline.ipynb`
3. 运行方案二：打开并依次执行 `Optiver_Scheme2_GroupTime_RF.ipynb`
4. 若存在 `test.csv`，Notebook 将在最后生成 `submission_scheme1.csv` 或 `submission_scheme2.csv`

## 方案说明
- 方案一（通用基线）：
  - 数值特征标准化、类别特征独热编码
  - 回归任务使用 Ridge；分类任务使用 LogisticRegression
  - 5 折交叉验证评估，逻辑清晰、可解释性强
- 方案二（分组时序增强）：
  - 按 `stock_id`、日期等分组构造统计与归一化特征（如组均值、相对值、日均值差）
  - 使用随机森林进行非线性拟合，增强鲁棒性与表达能力
  - 5 折交叉验证评估，适合复杂市场结构与横截面异质性

## 飞书页面内容（整理版）
- 来源链接：`https://dqpzque4vfe.feishu.cn/wiki/OlNPwx4uui6OflkB0D3c2iZOn2d`
- Optiver 介绍：
  - Optiver 是全球领先的做市商之一，专注于以技术驱动的量化交易，在收盘竞价等关键时刻提供流动性与价格发现。
  - Trading at the Close 任务聚焦收盘阶段的价格或方向预测，强调在订单簿结构变化与流动性波动下的稳定建模。
- 对应两个方案：
  - 方案一侧重稳健与可解释，适合快速建立基线、定位问题与开展误差分析。
  - 方案二通过分组与时序相关特征提升表达能力，适合更复杂的市场结构与横截面异质性场景。
- 建议：
  - 在上述基线之上可进一步引入 GBDT/Transformer、更细粒度时间窗特征与交易规则约束，以贴近真实决策流程。

## 注意事项
- 本仓库未包含比赛原始数据，请自行从 Kaggle 或内部数据源获取并放置到指定路径。
- 若你的数据列名不同（例如目标列、日期列等），Notebook 的自动检测会尝试兼容常见命名；如有偏差，请在 Notebook 中手动指定。

## 参考
- Kaggle 比赛页：`https://www.kaggle.com/competitions/optiver-trading-at-the-close`
- Optiver 公司介绍：公开资料与行业报道
