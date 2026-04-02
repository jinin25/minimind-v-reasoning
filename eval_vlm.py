import time
import argparse
import os
import re
import warnings
import torch
import random
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


class CaptureTextStreamer(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_parts = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.text_parts.append(text)
        super().on_finalized_text(text, stream_end=stream_end)

    def get_text(self):
        return ''.join(self.text_parts)


def clean_generated_text(text):
    text = text.replace('\r\n', '\n')
    text = re.sub(r'</?\s*think\s*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_reasoning_answer(text):
    text = clean_generated_text(text)
    reasoning_match = re.search(
        r'(?:^|\n)\s*Reasoning\s*[:：]\s*(.*?)(?=\n\s*Answer\s*[:：]|$)',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    answer_match = re.search(r'(?:^|\n)\s*Answer\s*[:：]\s*(.*)$', text, flags=re.IGNORECASE | re.DOTALL)

    reasoning_text = reasoning_match.group(1).strip() if reasoning_match else None
    answer_text = answer_match.group(1).strip() if answer_match else None
    return reasoning_text, answer_text, text


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        if args.weight == 'cot_vlm':
            ckp = os.path.expanduser(args.cot_weight_path)
        else:
            ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

        if not os.path.exists(ckp):
            raise FileNotFoundError(
                f"未找到权重文件: {ckp}\n"
                f"请确认路径后重试。示例命令:\n"
                f"1) python eval_vlm.py --load_from model --weight cot_vlm --device cuda\n"
                f"2) python eval_vlm.py --load_from model --weight cot_vlm --cot_weight_path /root/autodl-tmp/minimind-v/minimind-v/out/cot_vlm_768.pth --device cuda\n"
                f"3) python eval_vlm.py --load_from model --weight cot_vlm --cot_force_reasoning 0 --device cuda"
            )

        model = MiniMindVLM(
            VLMConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
            ),
            vision_model_path='./model/siglip2-base-p16-ve',
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniMindVLM.get_vision_model('./model/siglip2-base-p16-ve')

    get_model_params(model, model.config)
    preprocess = model.processor
    return model.half().eval().to(args.device), tokenizer, preprocess


def main():
    parser = argparse.ArgumentParser(
        description='MiniMind-V Chat',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            '说明:\n'
            '- cot_vlm 权重必须存在，否则会报错。\n'
            '- cot_force_reasoning 默认启用 Reasoning -> Answer 模板。\n'
            '- open_thinking 默认在 cot_vlm 下为 1，用户显式传参可覆盖。\n'
            '- cot_vlm 默认使用更稳的解码参数，用户传 --temperature/--top_p/--max_new_tokens 可覆盖。\n'
            '- 若输出解析不到 Reasoning/Answer，将 fallback 打印完整文本。\n\n'
            '示例:\n'
            '1. python eval_vlm.py --load_from model --weight cot_vlm --device cuda\n'
            '2. python eval_vlm.py --load_from model --weight cot_vlm --cot_weight_path /root/autodl-tmp/minimind-v/minimind-v/out/cot_vlm_768.pth --device cuda\n'
            '3. python eval_vlm.py --load_from model --weight cot_vlm --cot_force_reasoning 0 --device cuda'
        ),
    )
    parser.add_argument('--load_from', default='model', type=str, help='模型加载路径（model=原生torch权重，其他路径=transformers格式）')
    parser.add_argument('--save_dir', default='out', type=str, help='模型权重目录')
    parser.add_argument(
        '--weight',
        default='sft_vlm',
        type=str,
        choices=['pretrain_vlm', 'sft_vlm', 'cot_vlm'],
        help='权重名称前缀（pretrain_vlm, sft_vlm, cot_vlm）',
    )
    parser.add_argument(
        '--cot_weight_path',
        default='/root/autodl-tmp/minimind-v/minimind-v/out/cot_vlm_768.pth',
        type=str,
        help='cot_vlm 权重路径（仅 weight=cot_vlm 时生效）',
    )
    parser.add_argument(
        '--cot_force_reasoning',
        default=1,
        type=int,
        choices=[0, 1],
        help='cot_vlm 是否默认启用 Reasoning -> Answer 模板（0=否，1=是）',
    )
    parser.add_argument('--hidden_size', default=768, type=int, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否使用MoE架构（0=否，1=是）')
    parser.add_argument('--max_new_tokens', default=512, type=int, help='最大生成长度')
    parser.add_argument('--temperature', default=0.9, type=float, help='生成温度，控制随机性（0-1，越大越随机）')
    parser.add_argument('--top_p', default=0.9, type=float, help='nucleus采样阈值（0-1）')
    parser.add_argument('--image_dir', default='./dataset/eval_images/', type=str, help='测试图像目录')
    parser.add_argument('--show_speed', default=1, type=int, help='显示decode速度（tokens/s）')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='运行设备')
    parser.add_argument(
        '--open_thinking',
        default=None,
        type=int,
        choices=[0, 1],
        help='是否开启自适应思考（0=否，1=是，默认: cot_vlm=1，其他=0）',
    )
    parser.add_argument('--cot_temperature', default=0.35, type=float, help='cot_vlm 默认温度（未手动传 temperature 时生效）')
    parser.add_argument('--cot_top_p', default=0.8, type=float, help='cot_vlm 默认 top_p（未手动传 top_p 时生效）')
    parser.add_argument('--cot_max_new_tokens', default=256, type=int, help='cot_vlm 默认最大生成长度（未手动传 max_new_tokens 时生效）')
    args = parser.parse_args()

    is_cot_weight = args.weight == 'cot_vlm'
    effective_open_thinking = args.open_thinking if args.open_thinking is not None else (1 if is_cot_weight else 0)

    effective_temperature = args.temperature
    effective_top_p = args.top_p
    effective_max_new_tokens = args.max_new_tokens
    if is_cot_weight:
        if args.temperature == parser.get_default('temperature'):
            effective_temperature = args.cot_temperature
        if args.top_p == parser.get_default('top_p'):
            effective_top_p = args.cot_top_p
        if args.max_new_tokens == parser.get_default('max_new_tokens'):
            effective_max_new_tokens = args.cot_max_new_tokens

    model, tokenizer, preprocess = init_model(args)

    default_prompt = '请仔细观察这张图片，描述你所看到的内容：\n\n<image>'
    cot_prompt = (
        "Refer to following question and image.\n\n"
        "The question:\n"
        "Please describe the image content accurately. <image>\n\n"
        "Please reason step by step, and put your final answer (best option) within boxed{}."
        "Please follow these rules strictly:\n"
        "Output exactly in this format:\n"
        "</think>: 事实\n"
        "</answer>: 最终结论"
    )
    prompt = cot_prompt if (is_cot_weight and args.cot_force_reasoning) else default_prompt

    for image_file in sorted(os.listdir(args.image_dir)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            setup_seed(random.randint(1, 31415926))
            image_path = os.path.join(args.image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in MiniMindVLM.image2tensor(image, preprocess).items()}

            def build_inputs(prompt_text):
                messages = [
                    {
                        'role': 'user',
                        'content': prompt_text.replace('<image>', model.config.image_special_token * model.config.image_token_len),
                    }
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    open_thinking=bool(effective_open_thinking),
                )
                return tokenizer(text, return_tensors='pt', truncation=True).to(args.device)

            inputs = build_inputs(prompt)
            print(f'[图像]: {image_file}')
            print('💬: {}'.format(prompt.replace('\n', '\\n')))
            print('🤖: ', end='')
            st = time.time()

            if is_cot_weight:
                streamer = CaptureTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generated_ids = model.generate(
                    inputs=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=effective_max_new_tokens,
                    do_sample=True,
                    streamer=streamer,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=effective_top_p,
                    temperature=effective_temperature,
                    pixel_values=pixel_values,
                )

                final_text = streamer.get_text().strip()
                reasoning_text, answer_text, _ = parse_reasoning_answer(final_text)

                if reasoning_text is not None and answer_text is not None:
                    if reasoning_text.strip().lower() in {'n/a', 'na', 'none', '无', '空'}:
                        reasoning_text = '（模型未给出有效推理内容）'
                    print('\n[推理过程]')
                    print(reasoning_text if reasoning_text else '（模型未给出有效推理内容）')
                    print('\n[最终答案]')
                    print(answer_text if answer_text else '（模型未给出有效答案内容）')
                else:
                    print('\n[cot解析fallback] 未匹配到 Reasoning/Answer 模板，输出完整文本:')
                    print(clean_generated_text(final_text))
            else:
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generated_ids = model.generate(
                    inputs=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=effective_max_new_tokens,
                    do_sample=True,
                    streamer=streamer,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=effective_top_p,
                    temperature=effective_temperature,
                    pixel_values=pixel_values,
                )

            gen_tokens = len(generated_ids[0]) - len(inputs['input_ids'][0])
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')


if __name__ == '__main__':
    main()
