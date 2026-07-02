"""Freeze a stratified generation benchmark from the held-out General-SFT set."""
import json, re
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = "dataset/sft_i2t_val_1k.parquet"
OUTPUT = "dataset/general_generation_eval.parquet"
LIMITS = {"vqa": 100, "ocr": 100, "counting": 100, "short_answer": 100}

def category(question):
    q = question.lower()
    if re.search(r"ocr|文字|文本|写着|读出|单词|word|sign|标志牌|牌子", q): return "ocr"
    if re.search(r"how many|number of|多少|几个|数量|计数|count", q): return "counting"
    if re.search(r"选项|options?|颜色|color|什么类型|哪一|which|what is|是什么", q): return "short_answer"
    return "vqa"

t = pq.read_table(SOURCE)
selected, counts = [], {k: 0 for k in LIMITS}
for conv, image in zip(t["conversations"].to_pylist(), t["image_bytes"].to_pylist()):
    messages = json.loads(conv)
    q = next(x["content"] for x in messages if x["role"] == "user")
    a = next(x["content"] for x in messages if x["role"] == "assistant")
    kind = category(q)
    if counts[kind] >= LIMITS[kind]: continue
    selected.append((conv, image, kind, a)); counts[kind] += 1
out = pa.table({"conversations": [x[0] for x in selected], "image_bytes": [x[1] for x in selected],
                "category": [x[2] for x in selected], "reference": [x[3] for x in selected]})
pq.write_table(out, OUTPUT, compression="zstd")
report = {"source": SOURCE, "output": OUTPUT, "rows": len(selected), "categories": counts, "immutable_seed": "source-row-order-v1"}
Path("dataset/manifests/general_generation_eval_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
