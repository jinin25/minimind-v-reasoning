import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import glob
import hashlib
import io
import math
import re
import warnings
from bisect import bisect_right
from contextlib import nullcontext

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from model.model_vlm import MiniMindVLM, VLMConfig
from model.model_profiles import add_model_profile_argument, build_vlm_config
from reward.format import format_reward_fn
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    init_distributed_mode,
    init_vlm_model,
    is_main_process,
    setup_seed,
    vlm_checkpoint,
    vlm_collate_fn,
    load_vlm_weights,
)

warnings.filterwarnings("ignore")


def _stable_hash_percent(text, mod=10000):
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


def _safe_answer_to_str(answer):
    if isinstance(answer, list):
        if len(answer) == 0:
            return ""
        if len(answer) == 1:
            return str(answer[0])
        return ", ".join(str(x) for x in answer)
    if answer is None:
        return ""
    return str(answer)


class RLInnovatorVLDataset(Dataset):
    """Map-style RL dataset from RL_Innovator-VL parquet shards.

    This class only builds a lightweight (file_path, row_index) index,
    and lazily loads row-groups to keep memory usage low.
    """

    COLUMNS = [
        "id",
        "images",
        "problem",
        "answer",
        "problem_type",
        "answer_type",
        "source",
        "prompt_type",
    ]

    def __init__(
        self,
        data_dir,
        tokenizer,
        processor,
        split="train",
        val_ratio=0.02,
        split_seed="minimind-v-grpo",
        max_prompt_len=512,
        image_special_token="<|image_pad|>",
        image_token_len=64,
        enforce_answer_format=True,
        max_samples=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.split = split
        self.val_ratio = float(val_ratio)
        self.split_seed = split_seed
        self.max_prompt_len = max_prompt_len
        self.image_special_token = image_special_token
        self.image_token_len = image_token_len
        self.enforce_answer_format = bool(enforce_answer_format)

        self.shard_paths = sorted(glob.glob(os.path.join(data_dir, "RL_part*.parquet")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No RL shards found in: {data_dir}")

        self._parquet_files = {p: pq.ParquetFile(p) for p in self.shard_paths}
        self._row_group_offsets = {}
        for p, pf in self._parquet_files.items():
            offsets = []
            current = 0
            for rg_idx in range(pf.num_row_groups):
                offsets.append(current)
                current += pf.metadata.row_group(rg_idx).num_rows
            self._row_group_offsets[p] = offsets

        self._cached_key = None
        self._cached_table = None

        self.samples = []
        self._build_index(max_samples=max_samples)
        if len(self.samples) == 0:
            raise RuntimeError(f"Split={split} has no samples. Please check val_ratio/max_samples.")

    def _in_split(self, sample_id):
        if self.split == "all":
            return True
        if self.split == "val":
            threshold = int(self.val_ratio * 10000)
            key = f"{self.split_seed}-{sample_id}"
            return _stable_hash_percent(key) < threshold
        if self.split == "train":
            threshold = int(self.val_ratio * 10000)
            key = f"{self.split_seed}-{sample_id}"
            return _stable_hash_percent(key) >= threshold
        raise ValueError(f"Unsupported split: {self.split}")

    def _build_index(self, max_samples=0):
        kept = 0
        for path in self.shard_paths:
            pf = self._parquet_files[path]
            row_base = 0
            for batch in pf.iter_batches(batch_size=4096, columns=["id"]):
                ids = batch["id"].to_pylist()
                for i, sample_id in enumerate(ids):
                    sample_id = str(sample_id)
                    if self._in_split(sample_id):
                        self.samples.append((path, row_base + i, sample_id))
                        kept += 1
                        if max_samples > 0 and kept >= max_samples:
                            return
                row_base += len(ids)

    def __len__(self):
        return len(self.samples)

    def _load_row(self, path, row_idx):
        offsets = self._row_group_offsets[path]
        rg_idx = bisect_right(offsets, row_idx) - 1
        rg_idx = max(rg_idx, 0)
        cache_key = (path, rg_idx)
        if self._cached_key != cache_key:
            pf = self._parquet_files[path]
            self._cached_table = pf.read_row_group(rg_idx, columns=self.COLUMNS)
            self._cached_key = cache_key
        local_idx = row_idx - offsets[rg_idx]
        return self._cached_table, local_idx

    def _build_prompt(self, problem_text):
        image_tokens = self.image_special_token * self.image_token_len
        prompt_text = str(problem_text).replace("<image>", image_tokens).strip()
        if self.enforce_answer_format:
            prompt_text = (
                prompt_text
                + "\n\nPlease respond with exactly this structure:\n"
                + "<think>your reasoning</think>\n"
                + "<answer>final answer</answer>"
            )
        messages = [{"role": "user", "content": prompt_text}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_image_data(self, images_field):
        image_bytes = []
        for item in images_field or []:
            if isinstance(item, dict) and item.get("bytes") is not None:
                image_bytes.append(item["bytes"])

        if len(image_bytes) == 0:
            fallback_img = Image.new("RGB", (32, 32), color=(0, 0, 0))
            image_inputs_list = [MiniMindVLM.image2tensor(fallback_img, self.processor)]
        else:
            image_inputs_list = []
            for img_bytes in image_bytes:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image_inputs_list.append(MiniMindVLM.image2tensor(image, self.processor))

        if hasattr(image_inputs_list[0], "keys"):
            image_data = {
                k: torch.cat([inp[k] for inp in image_inputs_list], dim=0)
                for k in image_inputs_list[0].keys()
            }
        else:
            image_data = torch.stack(image_inputs_list)
        return image_data

    def __getitem__(self, idx):
        path, row_idx, sample_id = self.samples[idx]
        table, local_idx = self._load_row(path, row_idx)

        problem = table["problem"][local_idx].as_py()
        images = table["images"][local_idx].as_py()
        answer = table["answer"][local_idx].as_py()
        problem_type = table["problem_type"][local_idx].as_py()
        answer_type = table["answer_type"][local_idx].as_py()
        source = table["source"][local_idx].as_py()
        prompt_type = table["prompt_type"][local_idx].as_py()

        prompt = self._build_prompt(problem)
        pixel_values = self._build_image_data(images)

        return {
            "id": sample_id,
            "prompt": prompt,
            "raw_problem": str(problem),
            "answer": answer,
            "answer_type": str(answer_type),
            "problem_type": str(problem_type),
            "source": str(source),
            "prompt_type": str(prompt_type),
            "pixel_values": pixel_values,
        }


class VLMGRPOCollator:
    def __init__(self, tokenizer, max_prompt_len):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

    def __call__(self, batch):
        prompts = [x["prompt"] for x in batch]
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
            add_special_tokens=False,
            return_token_type_ids=False,
        )
        self.tokenizer.padding_side = old_padding_side

        fake_batch = [
            (tokenized.input_ids[i], tokenized.input_ids[i], x["pixel_values"])
            for i, x in enumerate(batch)
        ]
        _, _, pixel_values = vlm_collate_fn(fake_batch)

        return {
            "ids": [x["id"] for x in batch],
            "prompt_texts": prompts,
            "raw_problems": [x["raw_problem"] for x in batch],
            "answers": [x["answer"] for x in batch],
            "answer_types": [x["answer_type"] for x in batch],
            "problem_types": [x["problem_type"] for x in batch],
            "sources": [x["source"] for x in batch],
            "prompt_types": [x["prompt_type"] for x in batch],
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "pixel_values": pixel_values,
        }


class LocalRewarder:
    """组合格式、答案与可选 Judge 奖励。"""

    def __init__(
        self,
        format_weight=0.4,
        tag_weight=0.2,
        answer_weight=0.4,
        use_judge_reward=False,
        judge_weight=0.2,
    ):
        self.format_weight = float(format_weight)
        self.tag_weight = float(tag_weight)
        self.answer_weight = float(answer_weight)
        self.use_judge_reward = bool(use_judge_reward)
        self.judge_weight = float(judge_weight)
        self._judge_fn = None
        self._judge_warning_shown = False

        if self.use_judge_reward:
            self._try_load_judge()

    def _try_load_judge(self):
        try:
            from reward.custom_reward import compute_score

            self._judge_fn = compute_score
        except Exception as exc:
            self._judge_fn = None
            if not self._judge_warning_shown:
                Logger(f"[Reward] Judge disabled (import failed): {exc}")
                self._judge_warning_shown = True

    @staticmethod
    def _format_reward(completion):
        pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
        has_xml_format = 1.0 if re.match(pattern, completion, flags=re.S) else 0.0
        try:
            boxed_format = float(format_reward_fn(completion))
        except Exception:
            boxed_format = 0.0
        return max(has_xml_format, boxed_format)

    @staticmethod
    def _tag_count_reward(completion):
        score = 0.0
        if completion.count("<think>") == 1:
            score += 0.25
        if completion.count("</think>") == 1:
            score += 0.25
        if completion.count("<answer>") == 1:
            score += 0.25
        if completion.count("</answer>") == 1:
            score += 0.25
        return score

    @staticmethod
    def _answer_reward(completion, answer, answer_type):
        answer_str = _safe_answer_to_str(answer)
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.S | re.I)
        predicted = match.group(1).strip() if match else ""
        if not predicted or not answer_str:
            return 0.0
        t = str(answer_type).strip().lower()
        if "number" in t:
            pred_nums = re.findall(r"-?\d+(?:\.\d+)?", predicted)
            ref_nums = re.findall(r"-?\d+(?:\.\d+)?", answer_str)
            if not pred_nums or not ref_nums:
                return 0.0
            pred_value, ref_value = float(pred_nums[-1]), float(ref_nums[-1])
            tolerance = max(1e-6, abs(ref_value) * 1e-4)
            return float(abs(pred_value - ref_value) <= tolerance)

        normalize = lambda x: re.sub(r"\s+", " ", x).strip().lower()
        return float(normalize(predicted) == normalize(answer_str))

    def _judge_reward(self, completion, answer, prompt_text, problem_type, source, prompt_type):
        if self._judge_fn is None:
            return 0.0
        try:
            result = self._judge_fn(
                data_source=source,
                solution_str=completion,
                ground_truth=_safe_answer_to_str(answer),
                extra_info={
                    "question": prompt_text,
                    "problem_type": problem_type,
                    "prompt_type": prompt_type,
                    "data_source": source,
                },
            )
            return float(result.get("score", 0.0))
        except Exception as exc:
            if not self._judge_warning_shown:
                Logger(f"[Reward] Judge runtime error, fallback to local reward only: {exc}")
                self._judge_warning_shown = True
            return 0.0

    def score(self, completion, answer, answer_type, prompt_text, problem_type, source, prompt_type):
        format_score = self._format_reward(completion)
        tag_score = self._tag_count_reward(completion)
        answer_score = self._answer_reward(completion, answer, answer_type)

        reward = (
            self.format_weight * format_score
            + self.tag_weight * tag_score
            + self.answer_weight * answer_score
        )
        if self.use_judge_reward:
            reward += self.judge_weight * self._judge_reward(
                completion,
                answer,
                prompt_text,
                problem_type,
                source,
                prompt_type,
            )
        return float(reward)


def _to_device_pixel_values(pixel_values, device):
    if isinstance(pixel_values, dict):
        return {k: v.to(device) for k, v in pixel_values.items()}
    return pixel_values.to(device)


def _repeat_pixel_values(pixel_values, repeats):
    if isinstance(pixel_values, dict):
        return {k: torch.cat([v] * repeats, dim=0) for k, v in pixel_values.items()}
    return torch.cat([pixel_values] * repeats, dim=0)


def _build_full_attention_mask(attention_mask, gen_len):
    tail = torch.ones(
        (attention_mask.size(0), gen_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, tail], dim=1)


def _per_token_logps(model, input_ids, attention_mask, pixel_values, n_keep):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        logits_to_keep=n_keep + 1,
    )
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, -n_keep:]
    token_logps = torch.gather(logits.log_softmax(dim=-1), 2, targets.unsqueeze(-1)).squeeze(-1)
    aux_loss = out.aux_loss if out.aux_loss is not None else logits.new_zeros(())
    return token_logps, aux_loss


def _generate_k(model_for_gen, input_ids, attention_mask, pixel_values, tokenizer, args):
    outputs_list = []
    completion_ids_list = []
    for _ in range(args.num_generations):
        with torch.no_grad():
            outputs = model_for_gen.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion_ids = outputs[:, input_ids.size(1):]
        outputs_list.append(outputs)
        completion_ids_list.append(completion_ids)
    return outputs_list, completion_ids_list


def _compute_completion_mask(completion_ids, eos_token_id):
    if eos_token_id is None:
        return torch.ones_like(completion_ids, dtype=torch.int)
    is_eos = completion_ids.eq(eos_token_id)
    eos_idx = torch.full(
        (is_eos.size(0),),
        is_eos.size(1),
        dtype=torch.long,
        device=completion_ids.device,
    )
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (seq <= eos_idx.unsqueeze(1)).int()


def evaluate_reward(model, loader, tokenizer, rewarder, args, max_steps=20):
    model.eval()
    total_reward = 0.0
    total_count = 0

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if step > max_steps:
                break

            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            pixel_values = _to_device_pixel_values(batch["pixel_values"], args.device)

            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=args.max_gen_len,
                do_sample=False,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            completion_ids = outputs[:, input_ids.size(1):]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            for i, completion in enumerate(completions):
                r = rewarder.score(
                    completion=completion,
                    answer=batch["answers"][i],
                    answer_type=batch["answer_types"][i],
                    prompt_text=batch["prompt_texts"][i],
                    problem_type=batch["problem_types"][i],
                    source=batch["sources"][i],
                    prompt_type=batch["prompt_types"][i],
                )
                total_reward += r
                total_count += 1

    model.train()
    if total_count == 0:
        return 0.0
    return total_reward / total_count


def train_one_epoch(
    epoch,
    model,
    ref_model,
    loader,
    optimizer,
    scheduler,
    scaler,
    autocast_ctx,
    tokenizer,
    rewarder,
    vlm_config,
    args,
    start_step=0,
    wandb=None,
):
    use_scaler = scaler is not None and scaler.is_enabled()

    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        pixel_values = _to_device_pixel_values(batch["pixel_values"], args.device)

        model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
        outputs_list, completion_ids_list = _generate_k(
            model_for_gen=model_for_gen,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            tokenizer=tokenizer,
            args=args,
        )

        all_outputs = torch.cat(outputs_list, dim=0)
        all_completion_ids = torch.cat(completion_ids_list, dim=0)
        gen_len = all_completion_ids.size(1)

        full_attention = _build_full_attention_mask(attention_mask, gen_len)
        full_attention_rep = torch.cat([full_attention] * args.num_generations, dim=0)
        pixel_values_rep = _repeat_pixel_values(pixel_values, args.num_generations)

        with autocast_ctx:
            per_token_logps, aux_loss = _per_token_logps(
                model=model,
                input_ids=all_outputs,
                attention_mask=full_attention_rep,
                pixel_values=pixel_values_rep,
                n_keep=gen_len,
            )

        with torch.no_grad():
            ref_per_token_logps, _ = _per_token_logps(
                model=ref_model,
                input_ids=all_outputs,
                attention_mask=full_attention_rep,
                pixel_values=pixel_values_rep,
                n_keep=gen_len,
            )

        completions = tokenizer.batch_decode(all_completion_ids, skip_special_tokens=True)
        rewards_list = []
        for k in range(args.num_generations):
            for i in range(input_ids.size(0)):
                idx = k * input_ids.size(0) + i
                rewards_list.append(
                    rewarder.score(
                        completion=completions[idx],
                        answer=batch["answers"][i],
                        answer_type=batch["answer_types"][i],
                        prompt_text=batch["prompt_texts"][i],
                        problem_type=batch["problem_types"][i],
                        source=batch["sources"][i],
                        prompt_type=batch["prompt_types"][i],
                    )
                )
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=args.device)

        grouped_rewards = rewards.view(args.num_generations, -1).transpose(0, 1)
        mean_r = grouped_rewards.mean(dim=1, keepdim=True)
        std_r = grouped_rewards.std(dim=1, keepdim=True)
        advantages = (grouped_rewards - mean_r) / (std_r + 1e-4)
        advantages = torch.clamp(advantages, -10.0, 10.0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_flat = advantages.transpose(0, 1).reshape(-1)

        completion_mask = _compute_completion_mask(all_completion_ids, tokenizer.eos_token_id)
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1.0
        per_token_loss = -(
            torch.exp(per_token_logps - per_token_logps.detach()) * advantages_flat.unsqueeze(1)
            - args.beta * per_token_kl
        )
        policy_loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp_min(1)
        ).mean()

        loss = (policy_loss + args.aux_loss_coef * aux_loss) / args.accumulation_steps
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == len(loader):
            current_lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{len(loader)}), "
                f"policy_loss: {policy_loss.item():.4f}, "
                f"aux_loss: {float(aux_loss):.4f}, "
                f"reward: {rewards.mean().item():.4f}, "
                f"resp_len: {completion_mask.sum(dim=1).float().mean().item():.2f}, "
                f"lr: {current_lr:.8f}"
            )
            Logger(msg)

            if wandb and is_main_process():
                wandb.log(
                    {
                        "policy_loss": policy_loss.item(),
                        "aux_loss": float(aux_loss),
                        "reward": rewards.mean().item(),
                        "advantages_mean": advantages_flat.mean().item(),
                        "avg_response_len": completion_mask.sum(dim=1).float().mean().item(),
                        "learning_rate": current_lr,
                    }
                )

        if (step % args.save_interval == 0 or step == len(loader)) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if vlm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth"
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {
                k: v.half().cpu() for k, v in state_dict.items() if not k.startswith("vision_encoder.")
            }
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(
                vlm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=args.checkpoint_dir,
                scheduler=scheduler,
                scaler=scaler,
            )
            model.train()


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V GRPO training")
    add_model_profile_argument(parser)
    parser.add_argument("--data_dir", type=str, default="../dataset/RL_Innovator-VL", help="RL_Innovator-VL directory")
    parser.add_argument("--save_dir", type=str, default="../out", help="Path to save merged model weights")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="Path to save resume checkpoint")
    parser.add_argument("--save_weight", type=str, default="grpo_vlm", help="Checkpoint weight prefix")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=256)

    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.02, help="KL coefficient")
    parser.add_argument("--aux_loss_coef", type=float, default=1.0)

    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--val_steps", type=int, default=20)
    parser.add_argument("--split_seed", type=str, default="minimind-v-grpo")
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--max_val_samples", type=int, default=2000, help="0 means all")

    parser.add_argument("--format_weight", type=float, default=0.4)
    parser.add_argument("--tag_weight", type=float, default=0.2)
    parser.add_argument("--answer_weight", type=float, default=0.4)
    parser.add_argument("--use_judge_reward", action="store_true", help="Optional judge reward, default off")
    parser.add_argument("--judge_weight", type=float, default=0.2)

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--init_ckpt", type=str, default="../checkpoints/cot_vlm_768.pth", help="Initial SFT/VLM checkpoint")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="Resume from --save_weight checkpoint")
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="MiniMind-V-Reasoning")

    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    vlm_config = build_vlm_config(
        args.model_profile,
        args.max_prompt_len + args.max_gen_len,
        args.use_moe,
    )

    ckp_data = None
    if args.from_resume == 1:
        ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir=args.checkpoint_dir)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    if args.dtype == "float16":
        autocast_dtype = torch.float16
    elif args.dtype == "bfloat16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    if device_type == "cpu" or autocast_dtype == torch.float32:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.cuda.amp.autocast(dtype=autocast_dtype)

    scaler = None
    if device_type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    wandb = None
    if args.use_swanlab and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(
            project=args.swanlab_project,
            name=f"MiniMind-VLM-GRPO-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}",
            id=wandb_id,
            resume=("must" if wandb_id else None),
        )

    model, tokenizer, processor = init_vlm_model(
        vlm_config,
        from_weight="none",
        device=args.device,
        freeze_llm=0,
    )

    if ckp_data is None and args.init_ckpt and os.path.exists(args.init_ckpt):
        report = load_vlm_weights(model, args.init_ckpt, allow_vision_missing=True)
        Logger(
            f"Loaded init checkpoint: {args.init_ckpt}, "
            f"missing={len(report['missing'])}, unexpected={len(report['unexpected'])}"
        )
    elif ckp_data is None:
        Logger("No init checkpoint loaded. Training starts from current model state.")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    train_ds = RLInnovatorVLDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        processor=processor,
        split="train",
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        max_prompt_len=args.max_prompt_len,
        image_special_token=vlm_config.image_special_token,
        image_token_len=vlm_config.image_token_len,
        enforce_answer_format=True,
        max_samples=args.max_train_samples,
    )
    val_ds = RLInnovatorVLDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        processor=processor,
        split="val",
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        max_prompt_len=args.max_prompt_len,
        image_special_token=vlm_config.image_special_token,
        image_token_len=vlm_config.image_token_len,
        enforce_answer_format=True,
        max_samples=args.max_val_samples,
    )
    collate_fn = VLMGRPOCollator(tokenizer, max_prompt_len=args.max_prompt_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if dist.is_initialized() else None

    count_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    iters = len(count_loader)
    total_opt_steps = max(1, math.ceil(iters / args.accumulation_steps) * args.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data is not None:
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        if "scheduler" in ckp_data:
            scheduler.load_state_dict(ckp_data["scheduler"])
        if scaler is not None and "scaler" in ckp_data:
            scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data.get("epoch", 0)
        start_step = ckp_data.get("step", 0)
        Logger(f"Resume from epoch={start_epoch}, step={start_step}")

    ref_model = copy.deepcopy(model).to(args.device).eval().requires_grad_(False)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    rewarder = LocalRewarder(
        format_weight=args.format_weight,
        tag_weight=args.tag_weight,
        answer_weight=args.answer_weight,
        use_judge_reward=args.use_judge_reward,
        judge_weight=args.judge_weight,
    )

    Logger(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Iters/epoch: {iters}")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        setup_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        base_sampler = train_sampler if train_sampler is not None else indices
        batch_sampler = SkipBatchSampler(base_sampler, args.batch_size, skip)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        if skip > 0:
            Logger(f"Epoch[{epoch + 1}/{args.epochs}] resume: skip first {skip} steps")

        train_one_epoch(
            epoch=epoch,
            model=model,
            ref_model=ref_model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            autocast_ctx=autocast_ctx,
            tokenizer=tokenizer,
            rewarder=rewarder,
            vlm_config=vlm_config,
            args=args,
            start_step=skip,
            wandb=wandb,
        )

        start_step = 0

        if (epoch + 1) % args.val_interval == 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            val_reward = evaluate_reward(
                model=model,
                loader=val_loader,
                tokenizer=tokenizer,
                rewarder=rewarder,
                args=args,
                max_steps=args.val_steps,
            )
            Logger(f"[Validation] epoch={epoch + 1}, avg_reward={val_reward:.4f}")
            if wandb and is_main_process():
                wandb.log({"val_reward": val_reward, "epoch": epoch + 1})

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
