"""Audit pretraining parquet images and create a deterministic validation manifest."""

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def sample_id(conversations, image_value):
    images = image_value if isinstance(image_value, list) else [image_value]
    digest = hashlib.sha256(conversations.encode("utf-8"))
    for image_bytes in images:
        digest.update(hashlib.sha256(image_bytes).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="dataset/pretrain_i2t.parquet")
    parser.add_argument("--audit_samples", type=int, default=10000)
    parser.add_argument("--validation_samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--output_dir", default="dataset/manifests")
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.data_path)
    expected_schema = {"conversations", "image_bytes"}
    columns = set(parquet.schema_arrow.names)
    if columns != expected_schema:
        raise ValueError(f"Unexpected schema: {sorted(columns)}")

    row_count = parquet.metadata.num_rows
    rng = random.Random(args.seed)
    audit_indices = sorted(rng.sample(range(row_count), min(args.audit_samples, row_count)))
    validation_indices = sorted(rng.sample(range(row_count), min(args.validation_samples, row_count)))
    needed = set(audit_indices) | set(validation_indices)
    records = {}
    offset = 0

    for batch in parquet.iter_batches(columns=["conversations", "image_bytes"]):
        end = offset + batch.num_rows
        local_indices = [index - offset for index in needed if offset <= index < end]
        if local_indices:
            conversations = batch.column(0)
            images = batch.column(1)
            for local_index in local_indices:
                records[offset + local_index] = (
                    conversations[local_index].as_py(),
                    images[local_index].as_py(),
                )
        offset = end

    valid_images = 0
    invalid = []
    for index in audit_indices:
        _, image_value = records[index]
        images = image_value if isinstance(image_value, list) else [image_value]
        try:
            for image_bytes in images:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()
            valid_images += 1
        except Exception as exc:
            invalid.append({"row_index": index, "error": str(exc)})

    manifest = []
    for index in validation_indices:
        conversations, image_value = records[index]
        manifest.append({"row_index": index, "sample_id": sample_id(conversations, image_value)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pretrain_validation_v1.jsonl"
    with manifest_path.open("w", encoding="utf-8") as file:
        for row in manifest:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "data_path": str(Path(args.data_path).resolve()),
        "schema": parquet.schema_arrow.to_string(),
        "rows": row_count,
        "row_groups": parquet.num_row_groups,
        "audit_seed": args.seed,
        "audit_samples": len(audit_indices),
        "valid_samples": valid_images,
        "invalid_samples": len(invalid),
        "estimated_valid_rate": valid_images / len(audit_indices),
        "invalid_examples": invalid[:20],
        "validation_samples": len(manifest),
        "validation_manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    report_path = output_dir / "pretrain_audit_v1.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
