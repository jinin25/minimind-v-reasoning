import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import re
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model.model_vlm import MiniMindVLM
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

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
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2, cot_trim_ratio=0.3,
                         reasoning_drop_ratio=0.0):
    """训练时的数据增强：对 assistant 输出做随机变换

    1. 移除空 think 标签（80% 概率）
       目的：让模型学会在不需要推理时直接回答

    2. Reasoning Dropout：删除完整 think 块，只训练 answer

    3. 截断推理链到第一句话（30% 概率）
       目的：迫使模型用最少信息推理，适配 67M 小模型容量
    """
    # [增强 1] 80% 概率移除空 think 标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')

    # [增强 2] Reasoning Dropout。它与“截短推理”是两个不同实验变量。
    if reasoning_drop_ratio > 0 and random.random() < reasoning_drop_ratio:
        prompt_content = re.sub(
            r'<think>\s*.*?\s*</think>\s*', '', prompt_content,
            count=1, flags=re.DOTALL
        )

    # [增强 3] 30% 概率截断推理链到第一句话
    # 训练时随机截断，迫使模型学会"用最少推理步骤得出结论"
    if cot_trim_ratio > 0 and random.random() < cot_trim_ratio:
        match = re.search(r'<think>\s*(.*?)\s*</think>', prompt_content, re.DOTALL)
        if match:
            body = match.group(1).strip()
            # 取第一个句号/句点作为截断点
            for sep in ['。', '. ']:
                if sep in body:
                    body = body.split(sep)[0] + sep
                    break
            if len(body) > 5:  # 截断后仍有效才替换
                prompt_content = prompt_content[:match.start(1)] + body + prompt_content[match.end(1):]

    return prompt_content


class VLMDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='<|image_pad|>', image_token_len=64,
                 reasoning_drop_ratio=0.0, cot_trim_ratio=0.0,
                 enable_augmentation=True, answer_loss_weight=1.0):
        super().__init__()
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len
        self.reasoning_drop_ratio = float(reasoning_drop_ratio)
        self.cot_trim_ratio = float(cot_trim_ratio)
        self.enable_augmentation = bool(enable_augmentation)
        self.answer_loss_weight = float(answer_loss_weight)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.xml_token_ids = {
            tag: tokenizer(tag, add_special_tokens=False).input_ids
            for tag in ('<think>', '</think>', '<answer>', '</answer>')
        }

    @staticmethod
    def _find_sequence(values, pattern, start=0):
        if not pattern:
            return -1
        for i in range(start, len(values) - len(pattern) + 1):
            if values[i:i + len(pattern)] == pattern:
                return i
        return -1

    def generate_loss_weights(self, input_ids, labels):
        weights = [1.0] * len(input_ids)
        factor = self.answer_loss_weight
        if factor <= 1.0:
            return weights

        # XML tags are always emphasized.
        for pattern in self.xml_token_ids.values():
            pos = 0
            while True:
                pos = self._find_sequence(input_ids, pattern, pos)
                if pos < 0:
                    break
                for j in range(pos, min(pos + len(pattern), len(weights))):
                    if labels[j] != -100:
                        weights[j] = factor
                pos += len(pattern)

        # The complete answer block and the assistant EOS receive the same weight.
        open_ids = self.xml_token_ids['<answer>']
        close_ids = self.xml_token_ids['</answer>']
        begin = self._find_sequence(input_ids, open_ids)
        end = self._find_sequence(input_ids, close_ids, max(begin, 0))
        if begin >= 0 and end >= 0:
            end += len(close_ids)
            for j in range(begin, min(end, len(weights))):
                if labels[j] != -100:
                    weights[j] = factor
        pos = 0
        while True:
            pos = self._find_sequence(input_ids, self.eos_id, pos)
            if pos < 0:
                break
            for j in range(pos, min(pos + len(self.eos_id), len(weights))):
                if labels[j] != -100:
                    weights[j] = factor
            pos += len(self.eos_id)
        return weights

    def __len__(self):
        return len(self.table)

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

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        if not isinstance(image_bytes, list): image_bytes = [image_bytes]

        if self.enable_augmentation:
            conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        if self.enable_augmentation:
            prompt = post_processing_chat(
                prompt,
                reasoning_drop_ratio=self.reasoning_drop_ratio,
                cot_trim_ratio=self.cot_trim_ratio,
            )
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        loss_weights = self.generate_loss_weights(input_ids, labels)

        image_inputs_list = [MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes]
        if hasattr(image_inputs_list[0], 'keys'):
            image_data = {k: torch.cat([inp[k] for inp in image_inputs_list], dim=0) for k in image_inputs_list[0].keys()}
        else:
            image_data = torch.stack(image_inputs_list)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        result = (torch.tensor(input_ids, dtype=torch.long),
                  torch.tensor(labels, dtype=torch.long), image_data)
        if self.answer_loss_weight > 1.0:
            result += (torch.tensor(loss_weights, dtype=torch.float32),)
        return result

# 测试parquet数据读取和可视化
if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        t = pa.Table.from_batches(pq.ParquetFile(path).iter_batches()); fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            img_data = t['image_bytes'][i].as_py(); img_data = img_data[0] if isinstance(img_data, list) else img_data
            ax[i].imshow(Image.open(io.BytesIO(img_data))); ax[i].axis('off')
            ax[i].set_title(json.loads(t['conversations'][i].as_py())[1]['content'][:30], fontsize=8)
        out = path.replace('.parquet', '_preview.png'); plt.savefig(out); print(f'已保存{out}, 共{len(t)}条')
