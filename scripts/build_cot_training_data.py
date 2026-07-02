"""Build deterministic CoT train/validation data with 25% general-SFT replay."""

import argparse
import hashlib
import heapq
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def key(conversation):
    messages = json.loads(conversation)
    compact = [{"role": x.get("role"), "content": " ".join(x.get("content", "").split())} for x in messages]
    return hashlib.sha256(json.dumps(compact, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        data = batch.to_pydict()
        yield from zip(data["conversations"], data["image_bytes"])


def write(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"conversations": [x[0] for x in records], "image_bytes": [x[1] for x in records]}), path, compression="zstd")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cot", default="dataset/sft_i2t_cot_distilled_clean.parquet")
    p.add_argument("--general", default="dataset/sft_i2t.parquet")
    p.add_argument("--output", default="dataset/cot_sft_mix_25.parquet")
    p.add_argument("--cot_val", default="dataset/cot_sft_val_1k.parquet")
    p.add_argument("--general_eval", default="dataset/general_generation_eval_400.parquet")
    p.add_argument("--seed", type=int, default=20260702)
    args = p.parse_args()

    unique = {}
    for conv, image in rows(args.cot):
        unique.setdefault(key(conv), (conv, image))
    ordered = sorted(unique.items(), key=lambda x: hashlib.sha256(f"{args.seed}:{x[0]}".encode()).digest())
    cot_val = [record for _, record in ordered[:1000]]
    cot_train = [record for _, record in ordered[1000:]]
    forbidden = set(unique)  # replay must not duplicate any distilled conversation

    replay_needed = (len(cot_train) + 2) // 3  # replay / total = 25%
    candidates = []
    general_eval = []
    seen = set()
    for conv, image in rows(args.general):
        k = key(conv)
        if k in forbidden or k in seen:
            continue
        seen.add(k)
        rank = int.from_bytes(hashlib.sha256(f"{args.seed}:general:{k}".encode()).digest(), "big")
        item = (-rank, k, conv, image)
        if len(general_eval) < 400: heapq.heappush(general_eval, item)
        elif rank < -general_eval[0][0]: heapq.heapreplace(general_eval, item)
        # Keep evaluation and training selections disjoint using independent rank space.
        train_rank = int.from_bytes(hashlib.sha256(f"{args.seed}:replay:{k}".encode()).digest(), "big")
        train_item = (-train_rank, k, conv, image)
        if len(candidates) < replay_needed + 400: heapq.heappush(candidates, train_item)
        elif train_rank < -candidates[0][0]: heapq.heapreplace(candidates, train_item)
    eval_keys = {x[1] for x in general_eval}
    replay = [(x[2], x[3]) for x in sorted(candidates, reverse=True) if x[1] not in eval_keys][:replay_needed]
    general_eval = [(x[2], x[3]) for x in sorted(general_eval, reverse=True)]

    mixed = [(hashlib.sha256(f"mix:{args.seed}:{key(x[0])}".encode()).digest(), x) for x in cot_train + replay]
    mixed.sort(key=lambda x: x[0])
    write(args.output, [x[1] for x in mixed])
    write(args.cot_val, cot_val)
    write(args.general_eval, general_eval)
    report = {
        "source_cot_rows": pq.ParquetFile(args.cot).metadata.num_rows,
        "unique_cot_rows": len(unique), "duplicate_cot_rows": pq.ParquetFile(args.cot).metadata.num_rows - len(unique),
        "cot_validation_rows": len(cot_val), "cot_train_rows": len(cot_train),
        "general_replay_rows": len(replay), "mixed_train_rows": len(mixed),
        "general_replay_fraction": len(replay) / len(mixed), "seed": args.seed,
    }
    Path("dataset/manifests/cot_sft_mix_25_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
