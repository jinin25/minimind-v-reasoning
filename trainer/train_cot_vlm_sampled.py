import os
import sys
import io
import json
import random

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, get_model_params, vlm_checkpoint, SkipBatchSampler, vlm_collate_fn

warnings.filterwarnings('ignore')


def pre_processing_chat(conversations, add_system_ratio=0.2):
    if any(conv.get('tools') for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class CoTVLMDatasetSampled(Dataset):
    """
    带采样功能的 CoT VLM 数据集
    
    修改说明：
    - SFT 数据随机采样 sft_sample_size 条（默认 30000）
    - MMMU 数据全量使用，支持重复 mmmu_repeat 倍（默认 3）
    - 支持按目标比例混合 SFT 与 CoT（默认 SFT:CoT=6:4）
    - CoT 阶段按任务分流：VQA 使用 Reasoning，Caption 禁止 Reasoning 模板污染
    """
    
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='<|image_pad|>', image_token_len=64,
                 sft_sample_size=30000, mmmu_repeat=3,
                 cot_drop_ratio=0.1, target_sft_ratio=0.6, target_cot_ratio=0.4):
        super().__init__()
        
        self.sft_sample_size = sft_sample_size
        self.mmmu_repeat = mmmu_repeat
        self.cot_drop_ratio = cot_drop_ratio
        self.target_sft_ratio = target_sft_ratio
        self.target_cot_ratio = target_cot_ratio
        
        print(f"Loading {len(parquet_path)} parquet files...")
        print(
            f"SFT sample size: {sft_sample_size}, MMMU repeat: {mmmu_repeat}, "
            f"cot_drop_ratio: {cot_drop_ratio}, target ratio SFT:CoT={target_sft_ratio}:{target_cot_ratio}"
        )

        if self.target_sft_ratio <= 0 or self.target_cot_ratio <= 0:
            raise ValueError("target_sft_ratio and target_cot_ratio must be positive.")
        
        self.tables = []
        for i, path in enumerate(parquet_path):
            print(f"Loading file {i+1}/{len(parquet_path)}: {path}")
            self.tables.append(pa.Table.from_batches(pq.ParquetFile(path).iter_batches()))
            print(f"Loaded {len(self.tables[-1])} rows")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        
        # 构建采样混合计划
        self._build_mix_plan()
    
    def _build_mix_plan(self):
        """
        构建数据混合计划：
        - 假设 tables[0] 是 SFT 数据，tables[1] 是 MMMU 数据
        - SFT 数据随机采样 sft_sample_size 条
        - MMMU 数据全量重复 mmmu_repeat 倍
        - 最终混合比例：由 target_sft_ratio/target_cot_ratio 控制
        """
        self.mix_plan = []
        
        if len(self.tables) < 2:
            # 只有一个数据集，全部使用
            for i in range(len(self.tables[0])):
                self.mix_plan.append((0, i))
            random.shuffle(self.mix_plan)
            print(f"Only one dataset, using all {len(self.mix_plan)} samples")
            return
        
        sft_table = self.tables[0]  # SFT 数据
        mmmu_table = self.tables[1]  # MMMU (CoT) 数据
        
        sft_total = len(sft_table)
        mmmu_total = len(mmmu_table)
        
        print(f"Original SFT samples: {sft_total}")
        print(f"Original MMMU samples: {mmmu_total}")
        
        # SFT 采样：随机选取 sft_sample_size 条
        sft_indices = list(range(sft_total))
        random.shuffle(sft_indices)
        sft_sampled_indices = sft_indices[:min(self.sft_sample_size, sft_total)]
        sft_samples = [(0, idx) for idx in sft_sampled_indices]
        
        # MMMU 重复：全量重复 mmmu_repeat 倍
        mmmu_samples = []
        for repeat in range(self.mmmu_repeat):
            for idx in range(mmmu_total):
                mmmu_samples.append((1, idx))
        
        print(f"Sampled SFT samples: {len(sft_samples)}")
        print(f"Repeated MMMU samples: {len(mmmu_samples)} ({mmmu_total} x {self.mmmu_repeat})")
        
        # 计算目标混合比例
        # 当前 MMMU 样本数量
        current_mmmu = len(mmmu_samples)
        
        # 根据目标比例计算需要多少 SFT 样本
        target_sft_count = int(current_mmmu * (self.target_sft_ratio / self.target_cot_ratio))
        
        # 如果目标 SFT 数量超过采样数量，调整 MMMU 重复次数
        if target_sft_count > len(sft_samples):
            # 使用所有采样的 SFT，调整 MMMU 以保持比例
            actual_sft = len(sft_samples)
            target_mmmu_count = int(actual_sft * (self.target_cot_ratio / self.target_sft_ratio))
            
            if target_mmmu_count < len(mmmu_samples):
                # 随机采样 MMMU
                random.shuffle(mmmu_samples)
                mmmu_samples = mmmu_samples[:target_mmmu_count]
                print(
                    f"Adjusted MMMU samples to {len(mmmu_samples)} to maintain "
                    f"~{100 * self.target_cot_ratio / (self.target_sft_ratio + self.target_cot_ratio):.1f}% ratio"
                )
        else:
            # 随机采样 SFT 以匹配比例
            random.shuffle(sft_samples)
            sft_samples = sft_samples[:target_sft_count]
            print(
                f"Adjusted SFT samples to {len(sft_samples)} to maintain "
                f"~{100 * self.target_sft_ratio / (self.target_sft_ratio + self.target_cot_ratio):.1f}% ratio"
            )
        
        # 合并并打乱
        self.mix_plan = sft_samples + mmmu_samples
        random.shuffle(self.mix_plan)
        
        # 统计最终比例
        sft_count = sum(1 for ds_idx, _ in self.mix_plan if ds_idx == 0)
        mmmu_count = sum(1 for ds_idx, _ in self.mix_plan if ds_idx == 1)
        total_count = len(self.mix_plan)
        
        print(f"\n=== Final Dataset Stats ===")
        print(f"Total samples: {total_count}")
        print(f"SFT samples: {sft_count} ({100*sft_count/total_count:.1f}%)")
        print(f"MMMU (CoT) samples: {mmmu_count} ({100*mmmu_count/total_count:.1f}%)")
        print(f"===========================\n")
    
    def __len__(self):
        return len(self.mix_plan)
    
    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            content = turn['content'].replace('<image>', self.image_special_token) if turn.get('role') != 'system' else turn['content']
            messages.append({"role": turn['role'], "content": content})
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
    
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels
    
    def _strip_image_token(self, text):
        return text.replace('<image>', '').strip()

    def _extract_reasoning_answer(self, content):
        content = content.strip()
        if 'Answer:' in content:
            reasoning, answer = content.split('Answer:', 1)
            reasoning = reasoning.replace('Reasoning:', '').strip()
            answer = answer.strip()
        else:
            reasoning, answer = 'N/A', content
        return reasoning if reasoning else 'N/A', answer

    def _normalize_caption_text(self, content):
        content = content.strip()
        if 'Answer:' in content:
            content = content.split('Answer:', 1)[1].strip()
        if content.startswith('Reasoning:'):
            lines = [line for line in content.splitlines() if not line.strip().startswith('Reasoning:')]
            content = '\n'.join(lines).strip()
        return content

    # ===== CoT =====
    def process_conversations(self, conversations, task_type='Caption'):
        """
        在线任务分流：
        - VQA: 使用 Reasoning/Answer 模板，cot_drop_ratio 仅作用于 VQA
        - Caption: 强制任务前缀，禁止 Reasoning/Answer 模板污染
        """
        conversations = [dict(turn) for turn in conversations]

        if task_type == 'VQA':
            for turn in conversations:
                if turn.get('role') == 'user':
                    question = self._strip_image_token(turn.get('content', ''))
                    turn['content'] = f"<image> [Task: VQA] Question: {question}" if question else "<image> [Task: VQA] Question:"
                    break

            for turn in conversations:
                if turn.get('role') == 'assistant':
                    reasoning, answer = self._extract_reasoning_answer(turn.get('content', ''))
                    if random.random() < self.cot_drop_ratio:
                        turn['content'] = f'Answer: {answer}'
                    else:
                        turn['content'] = f'Reasoning: {reasoning}\nAnswer: {answer}'
                    break
        else:
            for turn in conversations:
                if turn.get('role') == 'user':
                    caption_instruction = self._strip_image_token(turn.get('content', ''))
                    if caption_instruction:
                        turn['content'] = f"<image> [Task: Caption]\n{caption_instruction}"
                    else:
                        turn['content'] = "<image> [Task: Caption]"
                    break

            for turn in conversations:
                if turn.get('role') == 'assistant':
                    caption_text = self._normalize_caption_text(turn.get('content', ''))
                    turn['content'] = caption_text if caption_text else 'A concise image description.'
                    break
        
        return conversations
    
    def __getitem__(self, index: int):
        ds_idx, row_idx = self.mix_plan[index]
        table = self.tables[ds_idx]
        conversations = json.loads(table['conversations'][row_idx].as_py())
        image_bytes = table['image_bytes'][row_idx].as_py()
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        
        # ds_idx == 1 表示 MMMU(VQA)，ds_idx == 0 表示 SFT(Caption)
        task_type = 'VQA' if ds_idx == 1 else 'Caption'
        conversations = self.process_conversations(conversations, task_type=task_type)
        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        
        image_inputs_list = [MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes]
        if hasattr(image_inputs_list[0], 'keys'):
            image_data = {}
            for k in image_inputs_list[0].keys():
                values = [inp[k] for inp in image_inputs_list]
                if k == 'spatial_shapes':
                    # spatial_shapes 特殊处理：确保正确的形状
                    converted = []
                    for v in values:
                        if isinstance(v, list):
                            # list -> tensor，并去除多余维度
                            v = torch.tensor(v, dtype=torch.long).squeeze()
                            if v.dim() == 1:  # [H, W] -> [[H, W]]
                                v = v.unsqueeze(0)
                        converted.append(v)
                    image_data[k] = torch.cat(converted, dim=0)
                else:
                    image_data[k] = torch.cat(values, dim=0)
        else:
            image_data = torch.stack(image_inputs_list)
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_data


def init_cot_vlm_model(vlm_config, base_weight='../out/sft_vlm_768.pth', tokenizer_path='../model',
                       vision_model_path='../model/siglip2-base-p16-ve', device='cuda',
                       freeze_llm=0, dtype=torch.float32):
    print(f"init_cot_vlm_model called with base_weight={base_weight}, device={device}")
    tokenizer_path = os.path.abspath(tokenizer_path)
    vision_model_path = os.path.abspath(vision_model_path)
    if base_weight != "none":
        if os.path.exists(base_weight):
            base_weight = base_weight
        elif os.path.exists(os.path.abspath(base_weight)):
            base_weight = os.path.abspath(base_weight)
        else:
            fallback_weight = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out", os.path.basename(base_weight.replace("\\", "/"))))
            if os.path.exists(fallback_weight):
                print(f"base_weight path not found, fallback to: {fallback_weight}")
                base_weight = fallback_weight
            else:
                print(f"Warning: base_weight not found: {base_weight}, fallback to training from scratch")
                base_weight = "none"
    print(f"Resolved paths - tokenizer: {tokenizer_path}, vision: {vision_model_path}, weights: {base_weight}")

    import json
    from tokenizers import Tokenizer
    print("Loading tokenizer...")
    tokenizer_file = os.path.join(tokenizer_path, 'tokenizer.json')
    tokenizer_config_file = os.path.join(tokenizer_path, 'tokenizer_config.json')

    with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    tokenizer_obj = Tokenizer.from_file(tokenizer_file)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)

    for key in ["bos_token", "eos_token", "pad_token", "unk_token"]:
        if key in config:
            setattr(tokenizer, key, config[key])
    if "chat_template" in config:
        tokenizer.chat_template = config["chat_template"]
    print("Tokenizer loaded successfully")

    print(f"Creating MiniMindVLM model with vision_model_path={vision_model_path}...")
    model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path)
    print("Model created successfully")
    import sys
    sys.stdout.flush()

    print(f"Checking if base_weight != 'none': {base_weight}")
    sys.stdout.flush()

    if base_weight != "none":
        print(f"Loading weights from {base_weight}...")
        weights = torch.load(base_weight, map_location="cpu")
        if isinstance(weights, dict) and "model" in weights and isinstance(weights["model"], dict):
            weights = weights["model"]
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        print(f"Weights loaded, missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")
        del weights

    print("Setting parameter gradients...")
    sys.stdout.flush()

    # 默认先冻结所有参数，仅保留 vision_proj 可训练
    for name, param in model.named_parameters():
        param.requires_grad = 'vision_proj' in name

    if freeze_llm == 0:
        # 完全可训练（视觉编码器除外）
        for name, param in model.named_parameters():
            if 'vision_encoder' not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        # 仅解冻首层 LLM + vision_proj
        for name, param in model.model.named_parameters():
            if 'layers.0.' in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        # 仅训练 vision_proj
        pass

    # CoT 阶段统一冻结视觉编码器
    for name, param in model.named_parameters():
        if 'vision_encoder' in name:
            param.requires_grad = False

    print("Parameter gradients set, calling get_model_params...")
    sys.stdout.flush()

    get_model_params(model, vlm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    print("Getting model processor...")
    preprocess = model.processor
    print(f"Got processor, moving model to device: {device}")

    # Move to device first
    model = model.to(device)
    print("Model moved to device successfully")

    # Convert trainable parts to training dtype, keep vision_encoder in float32
    for name, param in model.named_parameters():
        if 'vision_encoder' not in name and param.requires_grad:
            param.data = param.data.to(dtype)

    # Convert vision_proj to training dtype (it's trainable)
    if hasattr(model, 'vision_proj'):
        model.vision_proj = model.vision_proj.to(dtype)

    return model, tokenizer, preprocess


def train_epoch(epoch, loader, iters, model, optimizer, scaler, autocast_ctx, args, vlm_config,
                skipped_batches=0, full_epoch_iters=None, wandb=None):
    if full_epoch_iters is None:
        full_epoch_iters = iters + skipped_batches

    start_time = time.time()
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=1):
        step_in_epoch = skipped_batches + step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()} if isinstance(pixel_values, dict) else pixel_values.to(args.device)
        lr = get_lr(epoch * full_epoch_iters + step_in_epoch, args.epochs * full_epoch_iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        valid_label_count = (labels[:, 1:] != -100).sum().item()
        if valid_label_count == 0:
            Logger(f"Skip step {step_in_epoch}: no valid labels in batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        if not torch.isfinite(loss):
            Logger(f"Skip step {step_in_epoch}: non-finite loss={loss.detach().float().item():.6f}, valid_labels={valid_label_count}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.accumulation_steps == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        if step_in_epoch % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(1, step) * max(0, iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step_in_epoch}/{full_epoch_iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step_in_epoch % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step_in_epoch, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, pixel_values, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V CoT SFT (Sampled Version)")
    
    # 模型保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='cot_vlm_sampled', type=str, help="保存权重的前缀名")
    
    # ===== 新增采样参数 =====
    parser.add_argument("--sft_sample_size", type=int, default=30000, help="SFT 数据采样数量 (原始 582144 条)")
    parser.add_argument("--mmmu_repeat", type=int, default=2, help="MMMU 数据重复倍数 (原始 838 条)")
    parser.add_argument("--cot_drop_ratio", type=float, default=0.0, help="VQA 中丢弃 Reasoning 仅保留 Answer 的概率")
    parser.add_argument("--target_sft_ratio", type=float, default=0.6, help="目标 SFT 混合比例")
    parser.add_argument("--target_cot_ratio", type=float, default=0.4, help="目标 CoT(MMMU) 混合比例")
    
    # 训练参数 (可参数化)
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数 (默认 2)")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size (可参数化)")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率 (默认 2e-6)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="训练精度类型 (默认 float32)")
    
    # 其他训练参数
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # 模型结构参数
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=2048, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # 数据路径
    parser.add_argument("--data_path", nargs='+', type=str, default=['../dataset/sft_i2t.parquet', '../dataset/mmmu_sft.parquet'], help="训练数据路径")
    
    # 权重加载
    parser.add_argument('--from_weight', default='sft_vlm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--base_weight', default='../out/sft_vlm_768.pth', type=str, help="CoT训练初始化权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_llm', default=0, type=int, choices=[0, 1, 2], help="冻结策略（0=完全可训练，1=冻结+解冻第0层，2=完全冻结仅训练proj）")
    
    # 编译和日志
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-CoT-SFT-Sampled", help="wandb项目名")
    
    args = parser.parse_args()

    # 初始化分布式训练
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, 
                           max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe))
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # 设置数据类型 (默认 float32)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    if args.dtype == "bfloat16":
        train_dtype = torch.bfloat16
        autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif args.dtype == "float16":
        train_dtype = torch.float16
        autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        # 默认 float32
        train_dtype = torch.float32
        autocast_ctx = nullcontext()

    # 初始化模型
    model, tokenizer, preprocess = init_cot_vlm_model(
        vlm_config, 
        base_weight=args.base_weight, 
        device=args.device, 
        freeze_llm=args.freeze_llm, 
        dtype=train_dtype
    )
    
    # 可选的 torch.compile 加速
    if args.use_compile == 1:
        model = torch.compile(model)

    # 分布式训练包装
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 创建采样数据集
    Logger(f"Creating sampled dataset with SFT={args.sft_sample_size}, MMMU_repeat={args.mmmu_repeat}")
    train_dataset = CoTVLMDatasetSampled(
        args.data_path, 
        tokenizer, 
        preprocess=preprocess,
        max_length=args.max_seq_len,
        sft_sample_size=args.sft_sample_size,
        mmmu_repeat=args.mmmu_repeat,
        cot_drop_ratio=args.cot_drop_ratio,
        target_sft_ratio=args.target_sft_ratio,
        target_cot_ratio=args.target_cot_ratio
    )

    def build_train_loader(dataset, epoch, skip_batches=0):
        if dist.is_initialized():
            base_sampler = DistributedSampler(dataset, shuffle=True)
            base_sampler.set_epoch(epoch)
        else:
            base_sampler = RandomSampler(dataset)

        batch_sampler = SkipBatchSampler(base_sampler, args.batch_size, skip_batches=skip_batches)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=vlm_collate_fn
        )
        full_iters = (len(base_sampler) + args.batch_size - 1) // args.batch_size
        return loader, full_iters

    # 优化器和 scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype != "float32" and device_type == "cuda"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # 从 checkpoint 恢复（如果有）
    start_epoch = 0
    start_step = 0
    if ckp_data:
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)
        if 'optimizer' in ckp_data:
            optimizer.load_state_dict(ckp_data['optimizer'])
        if 'scaler' in ckp_data and scaler.is_enabled():
            scaler.load_state_dict(ckp_data['scaler'])
        Logger(f"Resumed from epoch {start_epoch}, step {start_step}")

    # wandb 初始化
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (
            f"MiniMind-V-CoT-SFT-Sampled-Epoch-{args.epochs}-"
            f"BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "sft_sample_size": args.sft_sample_size,
                "mmmu_repeat": args.mmmu_repeat,
                "cot_drop_ratio": args.cot_drop_ratio,
                "target_sft_ratio": args.target_sft_ratio,
                "target_cot_ratio": args.target_cot_ratio,
                "dtype": args.dtype,
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.num_hidden_layers,
                "max_seq_len": args.max_seq_len,
                "use_moe": args.use_moe,
                "freeze_llm": args.freeze_llm,
            }
        )

    # 打印训练配置
    Logger("=" * 50)
    Logger("Training Configuration (Sampled Version)")
    Logger("=" * 50)
    Logger(f"SFT Sample Size: {args.sft_sample_size}")
    Logger(f"MMMU Repeat: {args.mmmu_repeat}")
    Logger(f"CoT Drop Ratio (VQA only): {args.cot_drop_ratio}")
    Logger(f"Target Ratio SFT:CoT = {args.target_sft_ratio}:{args.target_cot_ratio}")
    Logger(f"Total Dataset Size: {len(train_dataset)}")
    Logger(f"Epochs: {args.epochs}")
    Logger(f"Batch Size: {args.batch_size}")
    Logger(f"Learning Rate: {args.learning_rate}")
    Logger(f"Data Type: {args.dtype}")
    Logger(f"Device: {args.device}")
    Logger("=" * 50)

    # 训练循环
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        skip_batches = start_step if epoch == start_epoch else 0
        train_loader, full_epoch_iters = build_train_loader(train_dataset, epoch, skip_batches=skip_batches)
        iters = len(train_loader)

        if iters == 0:
            Logger(f"Epoch {epoch + 1}: no remaining batches after skip={skip_batches}, move to next epoch")
            start_step = 0
            continue

        Logger(
            f"Epoch {epoch + 1}: skip_batches={skip_batches}, remaining_steps={iters}, "
            f"full_steps={full_epoch_iters}"
        )
        
        train_epoch(
            epoch, 
            train_loader, 
            iters, 
            model, 
            optimizer, 
            scaler, 
            autocast_ctx, 
            args, 
            vlm_config,
            skipped_batches=skip_batches,
            full_epoch_iters=full_epoch_iters,
            wandb=wandb
        )
        
        # 重置 start_step
        start_step = 0

    # 保存最终模型
    if is_main_process():
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        final_ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}_final.pth'
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        clean_state_dict = {
            key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
        }
        clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
        torch.save(clean_state_dict, final_ckp)
        Logger(f"Final model saved to {final_ckp}")

    if wandb:
        wandb.finish()

    Logger("Training completed!")
