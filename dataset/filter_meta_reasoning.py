#!/usr/bin/env python3
"""Filter meta-reasoning from distilled CoT parquet without loading it all in RAM."""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.I | re.S)

# A valid rationale should reason about the task content, not discuss an answer,
# reference answer, response, or requested format as an object.
RULES = {
    "reference_answer_en": re.compile(
        r"\b(?:the\s+)?reference\s+answer\b|\bground[ -]?truth\s+answer\b", re.I
    ),
    "answer_meta_en": re.compile(
        r"\b(?:the|this|that|given|provided)\s+answer\s+"
        r"(?:follows|is|was|provides?|describes?|explains?|matches?|aligns?|"
        r"reflects?|indicates?|states?|suggests?|addresses?|responds?|correctly|accurately)\b",
        re.I,
    ),
    "response_meta_en": re.compile(
        r"\b(?:the|this|that|given|provided)\s+(?:response|description)\s+"
        r"(?:is|was|provides?|describes?|explains?|matches?|aligns?|reflects?|"
        r"indicates?|states?|suggests?|addresses?|responds?|correctly|accurately)\b",
        re.I,
    ),
    "answer_meta_zh": re.compile(
        r"参考答案|标准答案|给定答案|提供的答案|上述答案|这个答案|该答案|"
        r"答案之所以|答案(?:准确|正确|符合|匹配|描述|解释|表明|指出)|"
        r"该回答|这个回答|上述回答|给出的回答|回答之所以|"
        r"回答(?:准确|正确|符合|匹配|描述|解释|表明|指出)"
    ),
    "format_meta": re.compile(
        r"\b(?:requested|required|specified)\s+(?:format|structure)\b|"
        r"(?:要求|指定|规定)的(?:格式|结构)", re.I
    ),
}


def classify_meta(think):
    return [name for name, pattern in RULES.items() if pattern.search(think)]


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run(args):
    source = pq.ParquetFile(args.input)
    schema = source.schema_arrow
    output = Path(args.output)
    report_path = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.unlink(missing_ok=True)

    counters = Counter()
    examples = {name: [] for name in RULES}
    writer = pq.ParquetWriter(temp, schema, compression=args.compression)
    started = time.time()
    try:
        for batch in source.iter_batches(batch_size=args.batch_size):
            data = batch.to_pydict()
            keep = []
            for i, raw in enumerate(data["conversations"]):
                counters["input_rows"] += 1
                try:
                    conv = json.loads(raw)
                    assistant = conv[1].get("content", "")
                    match = THINK_RE.search(assistant)
                except (json.JSONDecodeError, TypeError, IndexError, AttributeError):
                    match = None
                if not match:
                    counters["invalid_format"] += 1
                    continue
                think = " ".join(match.group(1).split())
                reasons = classify_meta(think)
                if reasons:
                    counters["filtered_meta"] += 1
                    for reason in reasons:
                        counters[f"rule_{reason}"] += 1
                        if len(examples[reason]) < args.example_count:
                            examples[reason].append(think[:300])
                    continue
                keep.append(i)
                counters["kept_rows"] += 1

            if keep:
                arrays = {
                    name: [data[name][i] for i in keep]
                    for name in schema.names
                }
                writer.write_table(pa.Table.from_pydict(arrays, schema=schema))
            if counters["input_rows"] % args.log_every < batch.num_rows:
                print(
                    f"[{time.strftime('%H:%M:%S')}] input={counters['input_rows']:,} "
                    f"kept={counters['kept_rows']:,} filtered={counters['filtered_meta']:,}",
                    flush=True,
                )
    finally:
        writer.close()

    check = pq.ParquetFile(temp)
    if check.schema_arrow != schema:
        raise RuntimeError("输出 schema 与输入不一致")
    if check.metadata.num_rows != counters["kept_rows"]:
        raise RuntimeError("输出行数校验失败")
    os.replace(temp, output)

    report = {
        "input": str(Path(args.input).resolve()),
        "output": str(output.resolve()),
        "created_at": time.strftime("%F %T"),
        "elapsed_sec": round(time.time() - started, 2),
        "counts": dict(counters),
        "keep_ratio": counters["kept_rows"] / max(counters["input_rows"], 1),
        "rules": {name: pattern.pattern for name, pattern in RULES.items()},
        "examples": examples,
        "validation": {
            "rows": check.metadata.num_rows,
            "row_groups": check.metadata.num_row_groups,
            "schema_match": True,
        },
    }
    atomic_json(report_path, report)
    print(json.dumps({"counts": report["counts"], "keep_ratio": report["keep_ratio"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="过滤蒸馏数据中的模板化元推理")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50000)
    parser.add_argument("--example-count", type=int, default=5)
    parser.add_argument("--compression", default="snappy")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
