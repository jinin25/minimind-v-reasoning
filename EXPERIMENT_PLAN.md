# MiniMind-V-Reasoning Experiment Plan

## Phase 0：训练前验收（已完成）

- [x] 抽检10,000条Pretrain图片，全部可解码
- [x] 固定1,024条验证样本及SHA-256样本ID，并从训练集排除
- [x] 给Pretrain/SFT增加`--max_steps`
- [x] 记录gradient norm、吞吐与显存
- [x] 4卡DDP运行100 step
- [x] 验证checkpoint在第50步停止并从第51步恢复到第100步

验收结果：无NaN、无DDP hang；DistributedSampler按rank分片；resume后step、loss与SwanLab run连续。稳态吞吐约360 samples/s，峰值显存2.42 GB/卡。

## Phase 1：Multimodal Pretrain（已完成）

- [x] 从`reason_768.pth`初始化
- [x] 冻结SigLIP，训练Vision Projector + LLM第0层
- [x] 1,273,674条训练样本，1 epoch，max length 360
- [x] 保存权重、resume checkpoint、配置与SHA-256
- [x] Real / Zero / Shuffled Image消融
- [x] 生成loss曲线

验收结果：训练loss稳定下降；Real Image loss 3.0470，Zero 3.7419，Shuffled 3.6754；权重已严格加载并完成消融。

## Phase 2：General VLM-SFT（已完成）

- [x] 建立固定1,000条SFT验证集并从训练数据排除
- [x] 生成30K工程冒烟集、300K主实验集和600K规模消融集
- [x] 将Pretrain的DDP屏障与NCCL稳定配置复用到SFT
- [x] 完成30K工程冒烟，不作为正式结论
- [x] 完成SFT-300K正式基线，1 epoch，max length 768
- [x] 完成SFT-600K规模消融
- [x] 从600K继续训练未见过的2,303,511条数据，完成分阶段全量SFT
- [x] 保存最终权重、resume checkpoint、配置、指标与SHA-256

验收结果：全量阶段完成143,970/143,970步，固定验证loss降至3.0263；256条消融中Real/Zero/Shuffled分别为3.0692/3.3161/3.3081，视觉语义依赖得到保留。

## Phase 3：CoT-SFT

- [ ] 先对全量SFT进行固定生成式基线评测，覆盖VQA、OCR、计数和短答案
- [ ] 对186,094条clean CoT去重，并混合20–30%普通SFT replay防止遗忘
- [ ] 运行100–500 step冒烟，确认1024长度下的batch、显存、格式与恢复链路
- [ ] 从同一全量SFT权重运行CoT-SFT（Reasoning Dropout=0）
- [ ] 运行严格同配置的Reasoning Dropout=0.2消融
- [ ] 比较推理准确率、普通VQA保持率、reasoning-on/off与格式合规率

建议正式配置：2 epochs、max length 1024、learning rate 1e-6～2e-6；先以实测显存确定有效全局batch 32或64。只有固定生成评测链路完成后才启动正式CoT对照，避免只凭loss判断推理能力。

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
