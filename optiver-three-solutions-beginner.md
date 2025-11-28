# Optiver 三套方案入门总结（Baseline / Robust Single / Top-15）

本指南面向小白，帮助快速理解三套常见方案在特征、目标、模型、训练与推断上的差异与联系。你不需要有量化背景，只要跟着“例子”看公式就能理解每个特征的含义与作用。

## 任务与指标
- 任务：根据每分钟订单簿数据（买卖价与数量、成交与不平衡等）预测当分钟的目标值 `target`。
- 指标：`MAE（平均绝对误差）`，越小越好。三套方案都以 MAE 为优化目标。

## 方案概览
- Baseline（多模型轻量融合）：以价量与不平衡的基础特征为主，训练 LGB / CatBoost（可选 XGB），推断时做均值融合。
- Robust Single（单模型稳健版）：在基础特征上增加交互与动量特征、严格的时间划分训练，单个 LightGBM 模型即可取得稳健效果。
- Top-15（系统强化版）：加入指数与板块对比、引领股票、Tick 推断价、滚动统计与更严谨的时序交叉验证，配合 LGB + XGB。

代码定位（便于你在仓库中对照）：
- Baseline 特征与训练：`baseline-lgb-xgb-and-catboost.ipynb:40-73`, `151-162`
- Robust Single 特征与训练：`optiver-robust-best-single-model.ipynb:883-964`, `1135-1177`
- Top-15 管线：`15th.py:230-276`, `278-323`, `338-496`, `541-566`, `596-758`

---

## 特征详解（含例子）

从四个维度理解与构造特征：时序、横截面、价量、不平衡，以及必要的预处理/映射。

### 一、时序特征（面向“过去发生了什么”）
常见形式：滞后值 `shift_k`、相对变化 `ret_k`、差分 `diff_k`、滚动统计（mean/median 等）。

- 滞后值 `shift_k`：取同一股票在过去第 `k` 步的某列值。
  - 例子：`reference_price_shift_3` 表示 3 分钟前的参考价。
- 相对变化 `ret_k`：当前与 `k` 分钟前的相对变化率（类似“收益率”）。
  - 例子：`wap_ret_2 = (wap_t - wap_{t-2}) / wap_{t-2}`，大于 0 表示两分钟内成交均价上行。
- 差分 `diff_k`：当前减去 `k` 分钟前的差值。
  - 例子：`bid_price_diff_1 = bid_price_t - bid_price_{t-1}`，正值代表买价上调。
- 滚动统计：在固定窗口（如 55 分钟）内对某列做 `mean/median/std` 等。
  - 例子：`rolling_nanmedian_revealed_target_55` 近似衡量上一日同时间段目标的中位水平（Top-15 使用“揭示目标”对齐上一日）。

代码参考：
- Baseline：`baseline-lgb-xgb-and-catboost.ipynb:75-79`
- Robust Single：`optiver-robust-best-single-model.ipynb:926-936`
- Top-15：`15th.py:311-323`, `419-466`

### 二、横截面特征（面向“当下市场的相对位置”）
在同一时间桶（`date_id` + `seconds_in_bucket`）上，比较个股与市场/板块平均或排名。

- 时间桶均值与排名：
  - 例子：`mean_wap_time_bucket` 是同一时间桶内所有股票的 `wap` 平均；`rank_wap_time_bucket` 是该股票在该时间桶的 `wap` 排名。
- 板块均值与情绪：
  - 例子：`sector_index` 为同板块在该时间桶的均价；`sector_imbalance` 为同板块买卖不平衡的均值。
- 相对表现：
  - 例子：`performance_over_index = 10000 * (wap - mean_wap_time_bucket)`，大于 0 代表该股强于市场平均。
- 引领股票透传：
  - 例子：在同时间桶将 MSFT/GOOGL/AMZN/NVDA/TSLA 的 `wap_ret_1` 等作为额外特征（这些“龙头”常率先反映市场变化）。

代码参考：`15th.py:364-377`, `481-496`

### 三、价量与不平衡特征（面向“订单簿结构与供需”）

核心概念：
- 成交均价 `wap`：用买卖量加权得到的“代表性价格”。
- 中间价 `mid_price = (ask_price + bid_price)/2`：价差的中心位置。
- 价差 `price_spread = ask_price - bid_price`：交易成本与波动空间的代理。
- 流动性不平衡 `liquidity_imbalance = (bid_size - ask_size)/(bid_size + ask_size)`：>0 表示买方更强。
- 成交与失衡比 `matched_imbalance = (imbalance_size - matched_size)/(matched_size + imbalance_size)`：供需压力的相对衡量。

小例子：
- 若 `bid_size=100`，`ask_size=50`，则 `liquidity_imbalance = (100-50)/(100+50) = 0.333`，买方更强；
- 若 `ask_price=1.0010`，`bid_price=0.9990`，则 `price_spread=0.0020`，价差较大，成交成本偏高。

尺度化不平衡（抵消量纲差异）：
- 两两不平衡 `a_b_imb = (a-b)/(a+b)`：
  - 例子：`far_price_near_price_imb = (far_price - near_price)/(far_price + near_price)`，大于 0 表示远端报价更高。
- 三元不平衡 `imb2 = (max - mid)/(mid - min)`：更敏感地捕捉极端偏向。
  - 例子：三价 `a=1.00, b=1.01, c=0.98`，`max=1.01, min=0.98, mid=1.00`，`imb2=(1.01-1.00)/(1.00-0.98)=0.01/0.02=0.5`，偏向上侧。

交互与衍生：
- 价格压力 `price_pressure = imbalance_size * price_spread`：失衡量遇到大价差，价格更易被推动。
- 市场紧迫度 `market_urgency = price_spread * liquidity_imbalance`：价差与偏向共同导致“急迫”交易环境。
- 深度压力 `depth_pressure = (ask_size - bid_size) * (far_price - near_price)`：深度差与远近价差的组合。
- 微观价格 `micro_price = ((bid_price * ask_size) + (ask_price * bid_size))/(bid_size + ask_size)`：考虑“对手方深度”的价格。

代码参考：
- Baseline（核心不平衡）：`baseline-lgb-xgb-and-catboost.ipynb:47-51`, `55-71`
- Robust Single（交互/动量扩展）：`optiver-robust-best-single-model.ipynb:889-919`
- Top-15（系统化）：`15th.py:287-305`, `306-309`, `299-304`

### 四、预处理与映射（让训练更稳、推断更快）

- 内存压缩 `reduce_mem_usage`：自动把大精度数值转为更小 dtype，提速同时省内存。
  - 例子：`float64 -> float32`，在本赛题足够且更省内存。
  - 代码参考：`15th.py:138-175`, `optiver-robust-best-single-model.ipynb:718-763`
- 股票全局统计映射 `global_stock_id_feats`：在训练集上统计每只股票的 `size/price` 的中位/标准差/极差，推断时直接按 `stock_id` 映射。
  - 作用：为个股提供“固有规模与波动”的先验。
  - 代码参考：`optiver-robust-best-single-model.ipynb:1072-1079`, `939-950`
- LookupInfo（Top-15）：将“历史标准差、时间桶指数统计、Tick 推断价中位数、反匿名 stock_info（sector 等）”打包持久化，推断时快速查询。
  - 作用：避免每次推断都重新扫描全量训练集，且让横截面/先验信息更丰富。
  - 代码参考：`15th.py:230-276`
- Tick 推断价（Top-15）：基于买价的最小跳动反推“价格缩放因子”，再派生推断价与体量/不平衡。
  - 例子：若某股票 55 分钟窗口内最小价格变动 `Δ=0.01`，则 `inferred_price=0.01/Δ`，用它去缩放现价与尺寸，得到更可比尺度。
  - 代码参考：`15th.py:397-417`
- 推断稳健化（Robust Single）：将预测序列做去均值、剪裁到区间，可选“零和约束”以消除系统性偏移。
  - 例子：`pred = pred - mean(pred)`；`clip(pred, -64, 64)`；可选 `zero_sum(pred, sqrt(volume))`。
  - 代码参考：`optiver-robust-best-single-model.ipynb:1265-1291`

---

## 目标与模型
- 目标：统一使用 MAE。
- 模型：
  - Baseline：`LGB (regression_l1)` + `CatBoost (MAE)`（可选 `XGB reg:absoluteerror`），5 折，推断均值融合。
  - Robust Single：`LGB (mae)` 单模型，时间划分严格避免泄漏，用 `best_iteration*1.2` 在全量训练上重训推断模型。
  - Top-15：`LGB (mae)` + `XGB (reg:quantileerror, alpha=0.5)`，更严谨的时序交叉验证（Purged KFold with Embargo）。

代码参考：
- Baseline：`baseline-lgb-xgb-and-catboost.ipynb:151-162`
- Robust Single：`optiver-robust-best-single-model.ipynb:1135-1177`
- Top-15：`15th.py:617-629`, `707-741`

---

## 训练与验证策略
- 随机 KFold 容易泄漏（时间序列）；更推荐：
  - Robust Single：按时间阈值 `split_day` 划分（train/valid/test），先在 valid 早停与调参，再在全量 train 重训推断模型。
  - Top-15：使用“按日分组 + 间隔/禁运”（Purged KFold with Embargo）严格控制标签泄漏与交叉污染。

代码参考：
- Robust Single：`optiver-robust-best-single-model.ipynb:1040-1047`, `1156-1166`, `1175-1177`
- Top-15：`15th.py:541-566`

---

## 推断与融合
- Baseline：迭代环境中逐批生成特征并做多模型均值融合，稳健且简单。
- Robust Single：单模型预测，结合“去均值/剪裁/零和约束”等后处理，降低系统偏移与尾部风险。
- Top-15：提供“已挑选模型清单”做线上融合接口（结合更丰富特征与严格 CV）。

代码参考：
- Baseline：`baseline-lgb-xgb-and-catboost.ipynb:231-239`
- Robust Single：`optiver-robust-best-single-model.ipynb:1273-1299`
- Top-15：`15th.py:106-115`

---

## 选型建议（给第一次上手的你）
- 资源有限/希望快速跑通：选 Robust Single，单模型稳健，训练与推断流程清晰。
- 需要一个对照基线：选 Baseline，结构最简，便于理解核心不平衡特征与融合。
- 追求成绩与稳健性：选 Top-15，全套特征 + 更严谨的 CV + 多模型融合，线上表现更扎实。

---

## 术语速览
- `wap`：加权成交均价；`bid/ask`：买/卖价；`spread`：价差；`mid_price`：中间价。
- `imbalance_size` / `imbalance_buy_sell_flag`：不平衡量与方向；`matched_size`：成交量。
- `shift/ret/diff`：滞后/相对变化/差分；`rolling`：滚动窗口统计。
- `sector_id`：板块；`lead stocks`：龙头股票（MSFT/GOOGL/AMZN/NVDA/TSLA）。
- `LookupInfo`：预先统计并持久化的全局映射，用于推断查表。

---

如果你希望把“Robust Single 的稳健推断后处理（去均值/剪裁/零和约束）”集成到 Baseline 或 Top-15 的推断脚本中，我可以进一步提供一个可复用的模块，并在当前仓库中接入。
