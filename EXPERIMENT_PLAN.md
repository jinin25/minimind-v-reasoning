# MiniMind-V-Reasoning Experiment Plan

## Phase 0：训练前验收

- 全量或大样本检查Pretrain图片有效率
- 固定验证集与样本ID
- 给Pretrain/SFT增加`--max_steps`
- 记录gradient norm、吞吐与显存
- 4卡DDP运行100 step
- 验证checkpoint中断恢复一致性

验收：无NaN、无DDP hang、四卡数据不重复、resume后step与loss连续。

## Phase 1：Multimodal Pretrain

- 初始化：`reason_768.pth`
- 冻结：SigLIP
- 训练：Vision Projector + LLM第0层
- 数据：1,274,698条Pretrain
- 初始计划：1 epoch，max length 360

验收：验证loss下降；真实图片优于空图；权重可独立加载。

## Phase 2：General VLM-SFT

- 从2.9M原始SFT中确定性分层抽样30–60万条
- 训练LLM + Projector，冻结SigLIP
- 1 epoch，max length 768

验收：视觉描述、OCR、计数和短问答均优于Pretrain模型。

## Phase 3：CoT-SFT

- 186,094条clean CoT
- 混合20–30%普通SFT防止遗忘
- 2 epochs，max length 1024
- 对比Reasoning Dropout 0与0.2

验收：推理任务提升；普通VQA下降受控；无思考模式仍能输出答案。

## Phase 4：Rule-based GRPO

- G0：1,000条可验证任务冒烟
- G1：5,000条低难度任务
- G2：10,000–20,000条混合任务
- 任务：选择题、数字、OCR、短字符串

验收：reward方差非零；退化组比例受控；答案准确率而非仅格式分提升；KL稳定。

## Phase 5：最终评估

逐阶段比较Pretrain、General SFT、CoT-SFT、Dropout和GRPO：

- 总体及分任务准确率
- reasoning-on/off
- 格式合规率
- 普通VQA保持率
- reward、KL与退化组
- 训练时间、吞吐、峰值显存
- 固定成功与失败案例
