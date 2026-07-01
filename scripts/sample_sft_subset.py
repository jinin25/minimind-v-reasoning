"""Create deterministic, disjoint SFT smoke and validation parquet subsets."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="dataset/sft_i2t.parquet")
    parser.add_argument("--train_output", default="dataset/sft_i2t_30k.parquet")
    parser.add_argument("--manifest", default="dataset/manifests/sft_smoke_30k_v1.json")
    parser.add_argument("--train_samples", type=int, default=30000)
    parser.add_argument("--validation_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.source)
    rows = parquet.metadata.num_rows
    rng = random.Random(args.seed)
    selected = rng.sample(range(rows), args.train_samples + args.validation_samples)
    train_indices = sorted(selected[: args.train_samples])
    validation_indices = sorted(selected[args.train_samples :])
    wanted = set(train_indices)
    pieces = []
    offset = 0
    for batch in parquet.iter_batches():
        end = offset + batch.num_rows
        local = [index - offset for index in wanted if offset <= index < end]
        if local:
            pieces.append(pa.Table.from_batches([batch]).take(pa.array(sorted(local))))
        offset = end
    table = pa.concat_tables(pieces)
    if table.num_rows != args.train_samples:
        raise RuntimeError(f"Expected {args.train_samples}, got {table.num_rows}")
    pq.write_table(table, args.train_output, compression="zstd", row_group_size=5000)

    manifest = {
        "source": args.source,
        "source_rows": rows,
        "seed": args.seed,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "train_indices_sha256": hashlib.sha256(json.dumps(train_indices).encode()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(json.dumps(validation_indices).encode()).hexdigest(),
        "validation_indices": validation_indices,
    }
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "validation_indices"}, indent=2))


if __name__ == "__main__":
    main()
