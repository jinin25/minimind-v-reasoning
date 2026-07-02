"""Build the SFT complement after the deterministic 600K subset and validation set."""

import argparse
import bisect
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="dataset/sft_i2t.parquet")
    parser.add_argument("--output", default="dataset/sft_i2t_remaining_2303k.parquet")
    parser.add_argument("--validation_manifest", default="dataset/manifests/sft_smoke_30k_v1.json")
    parser.add_argument("--excluded_train_samples", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--manifest", default="dataset/manifests/sft_remaining_after_600k_v1.json")
    args = parser.parse_args()

    source = pq.ParquetFile(args.source)
    total_rows = source.metadata.num_rows
    validation = set(json.loads(Path(args.validation_manifest).read_text(encoding="utf-8"))["validation_indices"])
    rng = random.Random(args.seed)
    trained = set()
    while len(trained) < args.excluded_train_samples:
        index = rng.randrange(total_rows)
        if index not in validation:
            trained.add(index)
    excluded = sorted(trained | validation)
    expected = total_rows - len(excluded)

    writer = None
    written = 0
    offset = 0
    for batch in source.iter_batches():
        table = pa.Table.from_batches([batch])
        end = offset + batch.num_rows
        left = bisect.bisect_left(excluded, offset)
        right = bisect.bisect_left(excluded, end)
        mask = np.ones(batch.num_rows, dtype=bool)
        for index in excluded[left:right]:
            mask[index - offset] = False
        kept = table.filter(pa.array(mask))
        if writer is None:
            writer = pq.ParquetWriter(args.output, kept.schema, compression="zstd")
        writer.write_table(kept, row_group_size=5000)
        written += kept.num_rows
        offset = end
    writer.close()
    if written != expected:
        raise RuntimeError(f"Expected {expected}, wrote {written}")

    report = {
        "source": args.source,
        "source_rows": total_rows,
        "excluded_train_samples": len(trained),
        "excluded_validation_samples": len(validation),
        "remaining_samples": written,
        "seed": args.seed,
        "excluded_indices_sha256": hashlib.sha256(json.dumps(excluded).encode()).hexdigest(),
    }
    Path(args.manifest).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
