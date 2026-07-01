#!/usr/bin/env python3
"""检查 reason_768.pth 与项目主线结构是否严格兼容。"""

import argparse
import os
import sys

import torch

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_DIR)

from model.model_profiles import build_vlm_config
from model.model_vlm import MiniMindVLM
from trainer.trainer_utils import load_vlm_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(PROJECT_DIR, 'out', 'reason_768.pth'))
    parser.add_argument('--model_profile', default='reason_vlm_109m')
    args = parser.parse_args()

    config = build_vlm_config(args.model_profile, max_seq_len=32)
    # 这里只检查 LLM 和 projector 结构，不加载视觉编码器。
    model = MiniMindVLM(config, vision_model_path=os.path.join(PROJECT_DIR, 'model', '__not_used__'))
    report = load_vlm_weights(model, args.checkpoint, allow_vision_missing=True)

    input_ids = torch.tensor([[1, 10, 20, 30]], dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids=input_ids)
    if not torch.isfinite(output.logits).all():
        raise RuntimeError('前向结果包含 NaN/Inf')

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    parameter_count = sum(value.numel() for value in checkpoint.values())
    print('VALIDATION PASSED')
    print(f'profile={args.model_profile}')
    print(f'checkpoint_tensors={len(checkpoint)}')
    print(f'checkpoint_parameters={parameter_count:,}')
    print(f'allowed_visual_missing={len(report["missing"])}')
    print(f'logits_shape={tuple(output.logits.shape)}')


if __name__ == '__main__':
    main()
