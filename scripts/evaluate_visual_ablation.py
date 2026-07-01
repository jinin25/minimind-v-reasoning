"""Compare validation loss with real images and zeroed images."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import VLMDataset
from model.model_profiles import build_vlm_config
from trainer.trainer_utils import PROJECT_DIR, init_vlm_model, vlm_collate_fn


def move_pixels(pixel_values, device):
    return {key: value.to(device) for key, value in pixel_values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="pretrain_vlm")
    parser.add_argument("--data_path", default=f"{PROJECT_DIR}/dataset/pretrain_i2t.parquet")
    parser.add_argument("--manifest", default=f"{PROJECT_DIR}/dataset/manifests/pretrain_validation_v1.jsonl")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=360)
    parser.add_argument("--sequential", action="store_true", help="Use the first N rows instead of a manifest")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=f"{PROJECT_DIR}/experiment_runs/p1_pretrain/visual_ablation.json")
    args = parser.parse_args()

    config = build_vlm_config("reason_vlm_109m", args.max_seq_len, 0)
    model, tokenizer, processor = init_vlm_model(
        config, from_weight=args.weight, device=args.device, freeze_llm=2
    )
    model.eval()
    dataset = VLMDataset(
        args.data_path,
        tokenizer,
        preprocess=processor,
        image_special_token=config.image_special_token,
        image_token_len=config.image_token_len,
        max_length=config.max_seq_len,
    )
    if args.sequential:
        indices = list(range(min(args.samples, len(dataset))))
    else:
        with open(args.manifest, "r", encoding="utf-8") as file:
            indices = [json.loads(line)["row_index"] for line in file if line.strip()][: args.samples]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=vlm_collate_fn,
    )

    real_total = zero_total = shuffled_total = 0.0
    seen = 0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for input_ids, labels, pixel_values in loader:
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            pixel_values = move_pixels(pixel_values, args.device)
            zeros = {key: torch.zeros_like(value) for key, value in pixel_values.items()}
            shuffled = {key: torch.roll(value, shifts=1, dims=0) for key, value in pixel_values.items()}
            real_loss = model(input_ids, labels=labels, pixel_values=pixel_values).loss
            zero_loss = model(input_ids, labels=labels, pixel_values=zeros).loss
            shuffled_loss = model(input_ids, labels=labels, pixel_values=shuffled).loss
            batch = input_ids.size(0)
            real_total += real_loss.item() * batch
            zero_total += zero_loss.item() * batch
            shuffled_total += shuffled_loss.item() * batch
            seen += batch

    result = {
        "samples": seen,
        "real_image_loss": real_total / seen,
        "zero_image_loss": zero_total / seen,
        "shuffled_image_loss": shuffled_total / seen,
        "zero_minus_real": (zero_total - real_total) / seen,
        "shuffled_minus_real": (shuffled_total - real_total) / seen,
        "uses_visual_signal": zero_total > real_total,
        "uses_matched_visual_semantics": shuffled_total > real_total,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
