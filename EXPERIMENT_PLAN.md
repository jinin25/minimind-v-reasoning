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

## Phase 3：CoT-SFT（已完成）

- [x] 建立固定General生成评测，覆盖VQA、OCR、计数和短答案
- [x] 对186,094条clean CoT去重，留出1,000条验证集并混入25%普通SFT replay
- [x] 完成两组500 step冒烟测试
- [x] 从同一全量SFT权重完成CoT-SFT（Reasoning Dropout=0）
- [x] 完成严格同配置的Reasoning Dropout=0.2消融
- [x] 比较固定验证loss、General保持、reasoning-on/off与格式合规率

建议正式配置：2 epochs、max length 1024、learning rate 1e-6～2e-6；先以实测显存确定有效全局batch 32或64。只有固定生成评测链路完成后才启动正式CoT对照，避免只凭loss判断推理能力。

结论：RD=0通用保持更好；RD=0.2的reasoning-on think完整率和reasoning-off CoT F1更高。保留两者，并以RD=0.2进入GRPO G0。

## Phase 4：Rule-based GRPO

- [ ] G0：从RD=0.2初始化，运行1,000条RL_Innovator-VL可验证任务冒烟（数据下载/启动中）
- G1：5,000条低难度任务
- G2：10,000–20,000条混合任务
- 任务：选择题、数字、OCR、短字符串

验收：reward方差非零；退化组比例受控；答案准确率而非仅格式分提升；KL稳定。

G0结束后先做同一固定RL验证集的GRPO前后比较。若reward仅由格式分驱动、答案准确率不升或KL异常，则停止扩量并修正reward；通过后才运行G1。

## Phase 5：最终评估

逐阶段比较Pretrain、General SFT、CoT-SFT、Dropout和GRPO：

- 总体及分任务准确率
- reasoning-on/off
- 格式合规率
- 普通VQA保持率
- reward、KL与退化组
- 训练时间、吞吐、峰值显存
- 固定成功与失败案例
