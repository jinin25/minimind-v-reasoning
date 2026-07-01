# MiniMind-V-Reasoning Experiment Report

本文只记录影响模型能力结论的主要实验与消融。环境校准、脚本修复、路径调整和冒烟测试不作为独立实验。

## 1. 实验总览

| 编号 | 实验 | 核心变量 | 状态 |
|---|---|---|---|
| E1 | CoT数据蒸馏 | 四卡异步教师生成 | 已完成 |
| E2 | CoT质量清洗 | 是否过滤模板化元推理 | 已完成 |
| E3 | Multimodal Pretrain | 真实图像 vs 全零图像 vs 错配图像 | 已完成 |
| E4 | General VLM-SFT | 30–60万分层数据 vs 全量数据 | 待实验 |
| E5 | CoT-SFT | 无CoT vs CoT；Dropout 0 vs 0.2 | 待实验 |
| E6 | Rule-based GRPO | SFT策略 vs GRPO策略 | 待实验 |

## 2. E1：CoT数据蒸馏

### 目的

从大规模图文SFT数据构造一句式结构化CoT，并将蒸馏控制在单机4卡约10小时内。

### 方法

- 教师：Qwen2.5-VL-7B-Instruct
- 部署：4张A10各运行一个vLLM实例
- 数据：确定性哈希抽样
- 输出：教师生成`<think>`，`<answer>`沿用原始答案
- 容错：每500条写Parquet分片和checkpoint
- 约束：推理语言与原始问答语言一致

### 结果

| 指标 | 数值 |
|---|---:|
| 原始行数 | 2,904,511 |
| 有效候选 | 2,202,245 |
| 候选池 | 360,000 |
| 请求尝试 | 301,551 |
| 成功样本 | 300,023 |
| Parse fail | 1,192 |
| Language fail | 336 |
| 成功率 | 99.49% |
| 最终吞吐 | 10.15 samples/s |
| 总耗时 | 8小时13分钟 |

### 结论

短输出、异步请求和四个独立vLLM实例适合单机多卡数据蒸馏；30万级样本可在一个工作日内完成。

## 3. E2：CoT质量清洗消融

### 目的

比较“仅保证XML格式”和“进一步过滤模板化元推理”的数据质量差异。

### 方法

只检查`<think>`内容，过滤“the answer follows”“reference answer”“该答案”等不分析问题本身的元叙述，不修改答案与图片。

### 结果

| 数据版本 | 样本数 | 元推理残留 | 格式/空答案/损坏图片 |
|---|---:|---:|---:|
| 格式清洗后 | 300,023 | 113,929 | 0 |
| 严格语义清洗后 | 186,094 | 0 | 0 |

严格清洗保留率为62.03%，另有71条重复conversation待训练采样时去重。

### 结论

格式正确不代表推理有效。后续CoT-SFT使用186,094条严格清洗数据，并将未严格清洗版本仅作为数据质量消融对照。

## 4. E3：Multimodal Pretrain

### 目的

在Reasoning LLM上建立视觉语言对齐，并通过图像置空消融验证模型确实使用视觉信息。

### 设置

| 项目 | 配置 |
|---|---|
| 初始化 | `reason_768.pth` |
| 数据 | 1,273,674训练 / 1,024固定验证 |
| 视觉编码器 | SigLIP P32/256，冻结 |
| 可训练模块 | Vision Projector + LLM第0层 |
| GPU | 4×A10 |
| Batch | 8/卡，global batch 32 |
| Sequence length | 360 |
| Epoch | 1 |
| Learning rate | 4e-4 |

### 训练结果

![Multimodal Pretrain loss curve](./experiment_runs/p1_pretrain/loss_curve.png)

| 指标 | 结果 |
|---|---:|
| Steps | 39,803 / 39,803 |
| Wall time | 1小时53分25秒 |
| 日志点 | 797 |
| 首个记录loss | 6.0462 |
| 最后记录loss | 3.1847 |
| 最后20点平均loss | 2.8587 |
| 最低记录loss | 2.1931 |
| 平均吞吐 | 189.45 samples/s |
| 峰值显存 | 2.44 GB/卡 |

曲线前期快速下降，约15k step后进入缓慢下降平台；后期batch波动存在，但移动平均没有反弹或发散。

### 视觉消融

在相同的256条固定验证样本上比较：

| 条件 | 输入 |
|---|---|
| Real Image | 原始图像 |
| Zero Image | shape相同的全零图像 |
| Shuffled Image | batch内循环错配的真实图像 |

| 条件 | Validation loss | 相对Real Image变化 |
|---|---:|---:|
| Real Image | 3.0470 | - |
| Zero Image | 3.7419 | +0.6949（+22.81%） |
| Shuffled Image | 3.6754 | +0.6284（+20.62%） |

### 结论

置空图和错配图均显著提高loss。错配图仍来自真实图像分布，因此结果不只是“全零像素异常”造成的惩罚，而表明模型利用了与文本匹配的视觉语义。该实验验证了视觉语言对齐，但不能替代SFT后的VQA/OCR/计数准确率评测。

### 产物

- 权重：`out/pretrain_vlm_768.pth`
- 权重SHA-256：`91a39c0b651ab6f5a7e89f4c9979d3a38f0250898d99c19a748444531d3493a4`
- Resume checkpoint：`checkpoints/pretrain_vlm_768_resume.pth`
- Resume SHA-256：`71622af1f87180b96de9f95fb880b31bec9638510d802a135fcee18daca47681`

## 5. E4：General VLM-SFT

计划对比30万、60万确定性分层样本；必要时补充290万全量1 epoch。统一从E3权重初始化，冻结SigLIP，训练LLM与Projector。主要指标为普通VQA、OCR、计数、图像描述和视觉置空下降幅度。

| 组别 | 数据量 | 作用 |
|---|---:|---|
| SFT-300K | 300,000 | 快速主实验 |
| SFT-600K | 600,000 | 数据规模消融 |
| SFT-Full | 2,904,511 | 资源允许时的上限对照 |

## 6. E5：CoT-SFT与Reasoning Dropout

| 组别 | CoT | Dropout | 目的 |
|---|---|---:|---|
| General SFT | 无 | 0 | 基线 |
| CoT-SFT | 有 | 0 | 测量CoT注入收益 |
| CoT-SFT + RD | 有 | 0.2 | 测量模板依赖与泛化变化 |

比较可验证推理准确率、普通VQA保持率、reasoning-on/off结果和格式合规率。

## 7. E6：Rule-based GRPO

从1,000条可验证任务开始，确认reward方差、KL和答案准确率变化后扩展到5,000及10,000–20,000条。最终对比同一CoT-SFT checkpoint在GRPO前后的准确率，而不是只比较格式奖励。

## 8. 最终结果表

| 模型阶段 | 普通VQA | OCR | 计数 | 推理准确率 | 置空下降 | 格式合规率 |
|---|---:|---:|---:|---:|---:|---:|
| Reason LLM | 待测 | 待测 | 待测 | 待测 | - | 待测 |
| + Pretrain | SFT后评测 | SFT后评测 | SFT后评测 | - | loss +20.62% | - |
| + General SFT | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| + CoT-SFT | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| + Reasoning Dropout | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| + GRPO | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
