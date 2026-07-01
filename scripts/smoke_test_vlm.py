"""Run one real-image forward/backward pass before starting distributed training."""

import argparse
import math
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import VLMDataset
from model.model_profiles import build_vlm_config
from trainer.trainer_utils import PROJECT_DIR, init_vlm_model, vlm_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=f"{PROJECT_DIR}/dataset/pretrain_i2t.parquet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_seq_len", type=int, default=360)
    args = parser.parse_args()

    config = build_vlm_config("reason_vlm_109m", args.max_seq_len, 0)
    model, tokenizer, processor = init_vlm_model(
        config,
        from_weight="reason",
        device=args.device,
        freeze_llm=1,
    )
    dataset = VLMDataset(
        args.data_path,
        tokenizer,
        preprocess=processor,
        image_special_token=config.image_special_token,
        image_token_len=config.image_token_len,
        max_length=config.max_seq_len,
    )
    input_ids, labels, pixel_values = vlm_collate_fn([dataset[0]])
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    pixel_values = {key: value.to(args.device) for key, value in pixel_values.items()}

    result = model(input_ids, labels=labels, pixel_values=pixel_values)
    loss = result.loss + result.aux_loss
    loss.backward()
    grad_sq = sum(
        parameter.grad.detach().float().norm().item() ** 2
        for name, parameter in model.named_parameters()
        if "vision_proj" in name and parameter.grad is not None
    )
    projector_grad_norm = math.sqrt(grad_sq)
    if not torch.isfinite(loss) or projector_grad_norm == 0:
        raise RuntimeError(f"Invalid smoke test: loss={loss.item()}, grad={projector_grad_norm}")
    print(f"SMOKE TEST PASSED loss={loss.item():.6f} projector_grad_norm={projector_grad_norm:.6f}")


if __name__ == "__main__":
    main()
