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
    parser.add_argument("--validation_output", default="dataset/sft_i2t_val_1k.parquet")
    parser.add_argument("--manifest", default="dataset/manifests/sft_smoke_30k_v1.json")
    parser.add_argument("--train_samples", type=int, default=30000)
    parser.add_argument("--validation_samples", type=int, default=1000)
    parser.add_argument("--fixed_validation_manifest", default="", help="Reuse validation_indices from an existing manifest")
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.source)
    rows = parquet.metadata.num_rows
    rng = random.Random(args.seed)
    if args.fixed_validation_manifest:
        fixed = json.loads(Path(args.fixed_validation_manifest).read_text(encoding="utf-8"))
        validation_indices = sorted(fixed["validation_indices"])
        validation_set = set(validation_indices)
        train_set = set()
        while len(train_set) < args.train_samples:
            index = rng.randrange(rows)
            if index not in validation_set:
                train_set.add(index)
        train_indices = sorted(train_set)
    else:
        selected = rng.sample(range(rows), args.train_samples + args.validation_samples)
        train_indices = sorted(selected[: args.train_samples])
        validation_indices = sorted(selected[args.train_samples :])
    train_pieces = []
    validation_pieces = []
    offset = 0
    for batch in parquet.iter_batches():
        end = offset + batch.num_rows
        batch_table = pa.Table.from_batches([batch])
        local_train = [index - offset for index in train_indices if offset <= index < end]
        local_validation = [index - offset for index in validation_indices if offset <= index < end]
        if local_train:
            train_pieces.append(batch_table.take(pa.array(local_train)))
        if local_validation:
            validation_pieces.append(batch_table.take(pa.array(local_validation)))
        offset = end
    table = pa.concat_tables(train_pieces)
    validation_table = pa.concat_tables(validation_pieces)
    if table.num_rows != args.train_samples:
        raise RuntimeError(f"Expected {args.train_samples}, got {table.num_rows}")
    pq.write_table(table, args.train_output, compression="zstd", row_group_size=5000)
    pq.write_table(validation_table, args.validation_output, compression="zstd", row_group_size=1000)

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
