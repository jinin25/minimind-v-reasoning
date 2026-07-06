import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import shutil
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig
from model.model_profiles import add_model_profile_argument, build_vlm_config
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import PROJECT_DIR, get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler, vlm_collate_fn

warnings.filterwarnings('ignore')


def sft_collate_fn(batch):
    base = vlm_collate_fn([item[:3] for item in batch])
    if len(batch[0]) == 4:
        return (*base, torch.stack([item[3] for item in batch]))
    return (*base, None)


@torch.no_grad()
def evaluate_validation(loader):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    for batch_index, (input_ids, labels, pixel_values, loss_weights) in enumerate(loader):
        if args.eval_batches > 0 and batch_index >= args.eval_batches:
            break
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        loss_weights = loss_weights.to(args.device) if loss_weights is not None else None
        pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()} if isinstance(pixel_values, dict) else pixel_values.to(args.device)
        with autocast_ctx:
            result = model(input_ids, labels=labels, pixel_values=pixel_values,
                           loss_weights=loss_weights)
            batch_loss = result.loss + result.aux_loss
        batch_size = input_ids.size(0)
        loss_sum += float(batch_loss) * batch_size
        sample_count += batch_size
    stats = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=args.device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    model.train()
    return float(stats[0] / stats[1])


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, val_loader=None):
    start_time = time.time()
    interval_time = start_time
    interval_samples = 0
    interval_loss_sum = 0.0
    interval_batches = 0
    grad_norm = 0.0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    for local_step, (input_ids, labels, pixel_values, loss_weights) in enumerate(loader):
        step = start_step + local_step
        completed_step = step + 1
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        loss_weights = loss_weights.to(args.device) if loss_weights is not None else None
        pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()} if isinstance(pixel_values, dict) else pixel_values.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        update_now = completed_step % args.accumulation_steps == 0 or completed_step == iters
        sync_ctx = model.no_sync() if isinstance(model, DistributedDataParallel) and not update_now else nullcontext()
        with sync_ctx:
            with autocast_ctx:
                res = model(input_ids, labels=labels, pixel_values=pixel_values,
                            loss_weights=loss_weights)
                batch_loss = res.loss + res.aux_loss
                loss = batch_loss / args.accumulation_steps
            scaler.scale(loss).backward()

        interval_samples += input_ids.size(0) * world_size
        interval_loss_sum += float(batch_loss.detach())
        interval_batches += 1
        if update_now:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        reached_limit = args.max_steps > 0 and completed_step >= args.max_steps
        if completed_step % args.log_interval == 0 or completed_step == iters or reached_limit:
            spend_time = time.time() - start_time
            loss_stats = torch.tensor([interval_loss_sum, interval_batches], dtype=torch.float64, device=args.device)
            if dist.is_initialized():
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            current_loss = float(loss_stats[0] / loss_stats[1])
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            interval_elapsed = max(time.time() - interval_time, 1e-6)
            samples_per_second = interval_samples / interval_elapsed
            memory_gb = torch.cuda.max_memory_allocated(args.device) / 1024 ** 3 if torch.cuda.is_available() else 0.0
            eta_min = spend_time / completed_step * max(iters - completed_step, 0) / 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({completed_step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, grad_norm: {grad_norm:.3f}, throughput: {samples_per_second:.2f} samples/s, peak_memory: {memory_gb:.2f} GB, eta: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "weighted_loss": current_logits_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "grad_norm": grad_norm, "throughput_samples_per_second": samples_per_second, "peak_gpu_memory_gb": memory_gb, "global_step": completed_step})
            interval_time = time.time()
            interval_samples = 0
            interval_loss_sum = 0.0
            interval_batches = 0

        if val_loader is not None and args.eval_interval > 0 and completed_step % args.eval_interval == 0:
            validation_loss = evaluate_validation(val_loader)
            Logger(f'Validation step {completed_step}: loss={validation_loss:.4f}')
            if wandb: wandb.log({"validation_loss": validation_loss, "global_step": completed_step})

        if (completed_step % args.save_interval == 0 or completed_step == iters or reached_limit) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}  # 半精度保存并移到CPU
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=completed_step, wandb=wandb, save_dir=args.checkpoint_dir, scaler=scaler)
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, pixel_values, res, loss
        if reached_limit:
            Logger(f'Reached --max_steps={args.max_steps}, stopping cleanly.')
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V SFT")
    add_model_profile_argument(parser)
    parser.add_argument("--save_dir", type=str, default=os.path.join(PROJECT_DIR, "out"), help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_vlm', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--max_steps", type=int, default=-1, help="最多训练多少个batch；-1表示完整epoch")
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(PROJECT_DIR, "checkpoints"), help="断点目录")
    parser.add_argument("--val_data_path", type=str, default="", help="固定验证集parquet")
    parser.add_argument("--eval_interval", type=int, default=500, help="每隔多少个micro-step验证；0关闭")
    parser.add_argument("--eval_batches", type=int, default=16, help="每次每个rank最多验证多少个batch；-1表示全量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default=os.path.join(PROJECT_DIR, "dataset", "sft_i2t.parquet"), help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain_vlm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_llm', default=0, type=int, choices=[0, 1, 2, 3], help="冻结策略（0=完全可训练，1=仅第0层，2=仅proj，3=顶部4层+proj）")
    parser.add_argument('--reasoning_drop_ratio', default=0.0, type=float, help="删除完整think块的概率；普通SFT设0，CoT-SFT建议0.2")
    parser.add_argument('--cot_trim_ratio', default=0.0, type=float, help="将推理截到第一句的概率，默认关闭")
    parser.add_argument('--answer_loss_weight', default=1.0, type=float,
                        help="XML标签、answer块和EOS的loss权重")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--use_swanlab", action="store_true", help="使用SwanLab记录实验")
    parser.add_argument("--swanlab_project", type=str, default="MiniMind-V-Reasoning", help="SwanLab项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = build_vlm_config(args.model_profile, args.max_seq_len, args.use_moe)
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir=args.checkpoint_dir) if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # SwanLab必须在DDP完成全部collective之后初始化。
    wandb = None

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device, freeze_llm=args.freeze_llm)
    train_ds = VLMDataset(
        args.data_path,
        tokenizer,
        preprocess=preprocess,
        image_special_token=vlm_config.image_special_token,
        image_token_len=vlm_config.image_token_len,
        max_length=vlm_config.max_seq_len,
        reasoning_drop_ratio=args.reasoning_drop_ratio,
        cot_trim_ratio=args.cot_trim_ratio,
        answer_loss_weight=args.answer_loss_weight,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    val_loader = None
    if args.val_data_path:
        val_ds = VLMDataset(
            args.val_data_path,
            tokenizer,
            preprocess=preprocess,
            image_special_token=vlm_config.image_special_token,
            image_token_len=vlm_config.image_token_len,
            max_length=vlm_config.max_seq_len,
            enable_augmentation=False,
        )
        val_sampler = DistributedSampler(val_ds, shuffle=False) if dist.is_initialized() else None
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
                                num_workers=0, pin_memory=True, collate_fn=sft_collate_fn)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )
    if args.use_swanlab and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-V-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.swanlab_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=sft_collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            stopped = train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, val_loader)
        else:
            stopped = train_epoch(epoch, loader, len(loader), 0, wandb, val_loader)
        if is_main_process() and not stopped:
            suffix = '_moe' if vlm_config.use_moe else ''
            src = os.path.join(args.save_dir, f'{args.save_weight}_{vlm_config.hidden_size}{suffix}.pth')
            dst = os.path.join(args.save_dir, f'{args.save_weight}_epoch{epoch + 1}_{vlm_config.hidden_size}{suffix}.pth')
            shutil.copy2(src, dst)
            resume_src = os.path.join(args.checkpoint_dir, f'{args.save_weight}_{vlm_config.hidden_size}{suffix}_resume.pth')
            resume_dst = os.path.join(args.checkpoint_dir, f'{args.save_weight}_epoch{epoch + 1}_{vlm_config.hidden_size}{suffix}_resume.pth')
            shutil.copy2(resume_src, resume_dst)
            Logger(f'Saved epoch snapshot: {dst}')
        if dist.is_initialized(): dist.barrier()
        if stopped:
            break

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
