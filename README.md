# 项目介绍

在原有[Minimind-v](https://github.com/jingyaogong/minimind-v#)基础上，增加**推理**功能，让模型具有思考能力！

# 快速开始



## 从0开始训练


### 1' 环境准备

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<details>
<summary>注：提前测试Torch是否可用cuda</summary>

```bash
import torch
print(torch.cuda.is_available())
```

如果不可用，请自行去[torch_stable](https://download.pytorch.org/whl/torch_stable.html)
下载whl文件安装。参考[链接](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

### 2' 数据下载

minimind-v 原项目数据集（Pretrain+SFT阶段）：


从下文提供的[数据集链接](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset)
下载所需内容并放到`./dataset`下。


Cot阶段数据集：

从[MMMU](https://huggingface.co/datasets/modelscope/MMMU-Reasoning-Distill-Validation)和[Share4oReasoning](https://huggingface.co/datasets/Share4oReasoning)
下载所需内容并放到`./dataset`下，并执行下述命令，将数据集清洗为parquet格式：

```bash 
python build_cot_sft.py --mmmu_path dataset/MMMU-Reasoning-Distill-Validation --share4o_path dataset/Share4oReasoning
```

<details>
<summary>下载pretrain/sft数据须知</summary>

Pretrain数据：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_i2t.parquet
```

SFT数据：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_i2t.parquet
```

建议预留~2GB空间存放数据集，若无多余空间存放pretrain数据，可尝试跳过pretrain训练步骤直接进行sft训练。

</details>





### 3' 开始训练

**3.1 预训练（学图像描述）**

```bash
# 基础训练命令（从LLM权重开始，仅训练vision_proj）
python train_pretrain_vlm.py --epochs 4 --from_weight llm
```

> 执行预训练，得到 `pretrain_vlm_*.pth` 作为预训练的输出权重（其中*为模型的dimension，默认为768）


**3.2 第一阶段监督微调（学看图对话方式）**

```bash
# 基础训练命令（从预训练权重开始，全参数微调）
python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm
```

> 执行监督微调，得到 `sft_vlm_*.pth` 作为指令微调的输出权重


**3.3 第二阶段监督微调（学习推理能力）**

```bash
# 基础训练命令（从第一阶段微调权重开始，全参数微调）
python train_cot_vlm.py --epochs 2 --from_weight sft_vlm
```


<details>
<summary>注：训练须知</summary>

**训练特性：**
- 支持断点续训：添加`--from_resume 1`参数可从上次中断处继续训练
- 支持GPU数量变化：续训时GPU数量改变会自动转换step
- 原子性保存：使用临时文件+替换机制，防止保存过程中断导致权重损坏
- 每次保存同时生成`out/**.pth`（模型权重）和`checkpoints/**_resume.pth`（训练状态）文件

```bash
# 训练中断后，使用相同命令并添加 --from_resume 1
python train_sft_vlm.py --epochs 4 --from_resume 1
```

**参数说明：**
- `--from_weight`: 基础权重名称（llm, pretrain_vlm, none等）
- `--save_weight`: 保存权重的前缀名
- `--from_resume`: 是否续训（0=从头开始，1=从检查点继续）
- `--freeze_llm`: 是否冻结LLM参数（仅pretrain使用）
- 更多可直接参考代码

</details>


---

### 4' 测试模型效果

确保需要测试的模型`*.pth`文件位于`./out/`目录下。

```bash
# 测试SFT模型（默认）
python eval_vlm.py --weight sft_vlm

# 测试Cot模型
python eval_vlm.py --weight cot_vlm

# 测试Pretrain模型
python eval_vlm.py --weight pretrain_vlm

```



------------------
缺陷不足：

用这么小的模型来做推理，本身就具有一定局限性，另外数据集大小太小，模型可以用于学习推理的样本量太少了，可以选择下载大一点的数据集做训练和测试，效果会好很多。