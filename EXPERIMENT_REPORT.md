# MiniMind-V-Reasoning Experiment Report

## 1. 报告状态

- 项目阶段：训练前工程验证完成
- 当前主线：`reason_vlm_109m`
- 正式Pretrain/SFT/GRPO：尚未开始
- 硬件：4×NVIDIA A10 24GB
- 监控：SwanLab 0.7.20，已登录

## 2. 实验一：CoT蒸馏吞吐优化

### 目的

从大规模图文SFT数据构造一句式结构化CoT，同时将任务控制在约10小时内。

### 方法

- 教师：Qwen2.5-VL-7B-Instruct
- 部署：4张A10各运行一个vLLM实例
- 数据：确定性哈希抽样
- 请求：每实例异步并发
- 输出：教师只生成think，answer使用原始已验证答案
- 容错：每500条保存Parquet分片和source index checkpoint
- 质量规则：格式、占位文本和语言一致性校验

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
| 基准吞吐 | 13.31 samples/s |
| 最终吞吐 | 10.15 samples/s |
| 总耗时 | 29,567秒，约8小时13分钟 |

### 结论

多实例数据并行显著优于将7B模型做张量并行。短输出与vLLM连续批处理使30万级蒸馏在单机4卡内可行。

## 3. 实验二：模板化元推理清洗

### 问题

全量审计发现部分think并未直接分析图像或问题，而是输出“the answer follows”“reference answer”“该答案”等元叙述。它们格式正确，但训练价值较低，并可能强化模板套话。

### 方法

只扫描think内容，按中英文元叙述规则流式过滤；不修改答案和图片字段。

### 结果

| 指标 | 数值 |
|---|---:|
| 清洗前 | 300,023 |
| 过滤 | 113,929 |
| 清洗后 | 186,094 |
| 保留率 | 62.03% |
| 规则残留 | 0 |
| 格式/空答案/空推理/图片异常 | 0 |
| 重复conversation | 71 |

### 结论

只检查XML标签会高估CoT质量。对think语义进行独立质量控制是CoT-SFT前的必要步骤。

## 4. 实验三：Reason checkpoint结构兼容

### 问题

原服务器默认结构为8层、KV heads 4、FFN 2432，并包含Q/K Norm；`reason_768.pth`实际为16层、KV heads 2、FFN 2048且没有Q/K Norm。直接宽松加载只能匹配约28%参数。

### 修改

- 建立唯一配置`reason_vlm_109m`
- 16层、hidden 768、8/2 attention heads、FFN 2048
- 关闭Q/K Norm
- 使用严格加载器，只允许缺少视觉模块参数

### 结果

| 指标 | 数值 |
|---|---:|
| Checkpoint tensors | 147 |
| Checkpoint parameters | 108,946,176 |
| Shape mismatch | 0 |
| Unexpected | 0 |
| 允许缺少 | Vision Projector |
| 文本前向 | 通过，logits `(1,4,6400)` |

### 结论

主干已经实现结构级兼容，不再依赖`strict=False`掩盖错误。项目模型参数量应统一描述为约109M。

## 5. 实验四：SigLIP P32视觉链路

### 问题

实际视觉模型为P32/256：`256/32=8`，因此输出64个patch token。原Projector写死按P16的256 token reshape，无法直接训练。

### 修改

- 使用AutoModel/AutoImageProcessor加载视觉模型
- 根据`image_size`和`patch_size`自动计算source tokens
- Projector检查source/target tokens是否整除

### 结果

| 指标 | 数值 |
|---|---:|
| Vision class | SiglipVisionModel |
| Image / Patch | 256 / 32 |
| Source / Target tokens | 64 / 64 |
| Projector merge | 1 |
| 单图loss | 9.7699，有限 |
| Projector gradient norm | 63.35，非零 |

### 结论

图像编码、视觉token注入、LLM forward、loss和backward链路已打通。

## 6. GRPO实现分析

当前GRPO为原生PyTorch实现：

1. 每个prompt在线采样4个completion；
2. 按格式、标签、答案和可选judge计算奖励；
3. 对同一prompt的奖励执行组内均值/标准差归一化；
4. Policy与冻结Reference Model计算token log-prob；
5. 使用相对优势与token级KL构造loss；
6. DDP在4张GPU上同步更新。

已修复：

- 不再使用`freeze_llm=2`冻结整个LLM；
- 从`<answer>`而非仅从`boxed`中提取答案；
- 数字任务使用数值容差；
- 字符串任务使用归一化exact match。

尚需补充：任务类型路由、部分奖励、reward variance、退化组比例和KL分项日志。

### 是否需要VERL

当前不需要。VERL适合多机、多角色资源编排、独立rollout集群和大规模异步采样；当前单机4卡、109M模型更适合保留可读的原生实现。后续如果rollout成为主要瓶颈，再将生成引擎抽象成独立backend，而不是现在就整体迁移。

## 7. 数据与资产状态

| 资产 | 状态 |
|---|---|
| `out/reason_768.pth` | 已验证 |
| `model/siglip2-base-p32-256-ve` | 已验证 |
| `dataset/pretrain_i2t.parquet` | 1,274,698行，可读 |
| `dataset/sft_i2t.parquet` | 2,904,511行，可读 |
| `dataset/sft_i2t_cot_distilled_clean.parquet` | 186,094行，可读 |
| SwanLab | 0.7.20，已登录 |

## 8. 当前结论与限制

训练前的权重、数据、视觉链路和监控资产已经到位。尚不能给出最终模型效果结论，因为固定验证集、四卡100-step测试和正式训练尚未完成。下一轮实验应优先建立评测基线，而不是立即开始长时间训练。
