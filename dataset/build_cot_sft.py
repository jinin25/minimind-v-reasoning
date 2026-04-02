import argparse
import json
import os
import re
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk
from tqdm import tqdm


# ------------------ regex ------------------
IMAGE_PAT = re.compile(r"<\s*image.*?>", re.I)
ANSWER_PAT = re.compile(r"(?:answer|答案)\s*[:：]\s*(.*)", re.I)
CHOICE_PAT = re.compile(r"\b([A-D])\b")


# ------------------ utils ------------------
def clean(x):
    return (x or "").replace("\r", "").strip()


def normalize_user(text):
    text = IMAGE_PAT.sub("<image>", text or "")
    return text if "<image>" in text else text + "\n<image>"


# ------------------ image ------------------
def get_image(sample, base):
    candidates = []

    if isinstance(sample.get("images"), list):
        candidates += sample["images"]

    for k in ["image"] + [f"image_{i}" for i in range(1, 8)]:
        if k in sample:
            candidates.append(sample[k])

    imgs = []
    for c in candidates:
        try:
            if isinstance(c, dict) and "bytes" in c:
                b = c["bytes"]
            elif isinstance(c, str):
                path = c if os.path.isabs(c) else os.path.join(base, c)
                with open(path, "rb") as f:
                    b = f.read()
            else:
                continue

            if b and b not in imgs:
                imgs.append(b)
        except:
            continue

    if not imgs:
        return None, "image_decode_failed"
    if len(imgs) > 1:
        return None, "multi_image"

    return imgs[0], "ok"


# ------------------ text ------------------
def get_text(sample):
    # unified parsing
    for key in ["conversations", "messages"]:
        turns = sample.get(key)
        if not isinstance(turns, list):
            continue

        for t in turns:
            role = t.get("role") or t.get("from")
            content = t.get("content") or t.get("value")

            if role in ["user", "human"]:
                user = content
            elif role in ["assistant", "gpt"]:
                assistant = content

            if user and assistant:
                return user, assistant

    return None, None


# ------------------ answer ------------------
def parse_answer(text, gt, is_mc):
    text = text or ""

    # reasoning
    reasoning = re.sub(r"</?think>|</?answer>", "", text, flags=re.I)
    reasoning = re.sub(ANSWER_PAT, "", reasoning).strip() or "N/A"

    # answer
    sources = [gt, text]

    if is_mc:
        for s in sources:
            if not s:
                continue
            m = CHOICE_PAT.findall(s)
            if m:
                return f"Reasoning: {reasoning}\nAnswer: {m[-1]}", m[-1]
        return "", None

    ans = next((clean(s) for s in sources if s), "")
    return f"Reasoning: {reasoning}\nAnswer: {ans}", ans


# ------------------ main ------------------
def process(ds, base, writer):
    stats = Counter()

    for sample in tqdm(ds):
        stats["total"] += 1

        # image
        img, state = get_image(sample, base)
        if state != "ok":
            stats[f"drop_{state}"] += 1
            continue

        # text
        user, assistant = get_text(sample)
        if not user:
            stats["drop_missing_user"] += 1
            continue
        if not assistant:
            stats["drop_missing_assistant"] += 1
            continue

        # answer
        is_mc = bool(sample.get("options"))
        out, ans = parse_answer(assistant, sample.get("answer"), is_mc)

        if not ans:
            stats["drop_answer_normalize_failed"] += 1
            continue

        conv = json.dumps([
            {"role": "user", "content": normalize_user(user)},
            {"role": "assistant", "content": out},
        ], ensure_ascii=False)

        writer.write_table(pa.Table.from_pydict({
            "conversations": [conv],
            "image_bytes": [img]
        }))

        stats["keep"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share4o_path")
    parser.add_argument("--mmmu_path")
    parser.add_argument("--out1")
    parser.add_argument("--out2")
    args = parser.parse_args()

    schema = pa.schema([
        ("conversations", pa.string()),
        ("image_bytes", pa.binary())
    ])

    for name, path, out, split in [
        ("share4o", args.share4o_path, args.out1, "train"),
        ("mmmu", args.mmmu_path, args.out2, "validation"),
    ]:
        ds = load_from_disk(path)[split]

        writer = pq.ParquetWriter(out, schema)

        stats = process(ds, path, writer)
        writer.close()

        print(name, stats)


if __name__ == "__main__":
    main()