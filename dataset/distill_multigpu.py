#!/usr/bin/env python3
"""Resumable multi-endpoint CoT distillation for sft_i2t parquet."""

import argparse
import asyncio
import hashlib
import heapq
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI


def log(message):
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def extract_answer(text):
    text = text if isinstance(text, str) else ""
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()
    match = re.search(r"(?:答案|Answer)\s*[:：]\s*(.+)", text, re.S | re.I)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def parse_conversation(raw):
    try:
        conv = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(conv, list) or len(conv) < 2:
        return None
    user = conv[0].get("content", "") if isinstance(conv[0], dict) else ""
    answer = conv[1].get("content", "") if isinstance(conv[1], dict) else ""
    user = re.sub(r"<image>\s*", "", user if isinstance(user, str) else "").strip()
    answer = extract_answer(answer)
    if not user or not answer or len(user) < 10:
        return None
    if "?" not in user and "？" not in user and len(user) <= 50:
        return None
    return user, answer


def stable_score(index, seed):
    digest = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def select_candidates(input_path, cache_path, count, seed):
    cache_path = Path(cache_path)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("count") == count and payload.get("seed") == seed:
            indices = payload["indices"]
            log(f"复用候选索引: {len(indices):,} 条")
            return indices

    parquet = pq.ParquetFile(input_path)
    heap = []
    total = eligible = 0
    started = time.time()
    for batch in parquet.iter_batches(batch_size=10000, columns=["conversations"]):
        for raw in batch.column(0).to_pylist():
            index = total
            total += 1
            if parse_conversation(raw) is None:
                continue
            eligible += 1
            score = stable_score(index, seed)
            item = (-score, index)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        if total % 500000 == 0:
            log(f"扫描 {total:,} 行，有效 {eligible:,} 行")
    indices = sorted(index for _, index in heap)
    atomic_json(cache_path, {
        "input": str(input_path), "seed": seed, "count": count,
        "total_rows": total, "eligible_rows": eligible, "indices": indices,
    })
    log(f"候选抽样完成: {len(indices):,}/{eligible:,}，耗时 {time.time()-started:.1f}s")
    return indices


def load_completed(parts_dir):
    completed = set()
    for path in sorted(Path(parts_dir).glob("part-*.indices.json")):
        completed.update(json.loads(path.read_text(encoding="utf-8")))
    return completed


def prompt_for(question, answer):
    return (
        "Given the question and its verified reference answer, write exactly ONE concise "
        "sentence explaining why the answer follows. Wrap that actual explanatory sentence "
        "between an opening <think> tag and a closing </think> tag, with no other text. "
        "Never output placeholder words such as 'reasoning', 'explanation', or '推理'. "
        "Use the same language as the question. Do not repeat the answer verbatim and do not "
        "output answer tags.\n\n"
        f"Question:\n{question[:3500]}\n\nReference answer:\n{answer[:3500]}"
    )


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.S | re.I)


def parse_think(text):
    match = THINK_RE.search(text or "")
    if not match:
        return None
    think = " ".join(match.group(1).split()).strip()
    if not 8 <= len(think) <= 500:
        return None
    if "<answer" in think.lower() or "</think" in think.lower():
        return None
    normalized = re.sub(r"[\s.。!！?？_-]+", "", think).lower()
    placeholders = {"reasoning", "explanation", "推理", "解释", "onesentence", "一句话"}
    if normalized in placeholders or len(set(normalized)) < 4:
        return None
    return think


def language_matches(question, think):
    question_cjk = len(re.findall(r"[\u3400-\u9fff]", question))
    think_cjk = len(re.findall(r"[\u3400-\u9fff]", think))
    question_latin = len(re.findall(r"[A-Za-z]", question))
    if question_cjk >= 2:
        return think_cjk >= 2
    if question_latin >= 5 and question_cjk == 0:
        return think_cjk == 0
    return True


async def call_teacher(client, args, question, answer):
    for attempt in range(args.retries + 1):
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "Follow the requested output format exactly."},
                    {"role": "user", "content": prompt_for(question, answer)},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            raw = response.choices[0].message.content or ""
            think = parse_think(raw)
            if think and language_matches(question, think):
                return think, "ok"
            if attempt >= args.retries:
                return None, "language_fail" if think else "parse_fail"
        except Exception as exc:
            if attempt >= args.retries:
                return None, f"api_fail:{type(exc).__name__}"
            await asyncio.sleep(min(2 ** attempt, 8))


def build_output_row(row, user, answer, think, schema):
    conv = [
        {"role": "user", "content": "<image>\n" + user.replace("<image>", "").strip()},
        {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n<answer>\n{answer}\n</answer>"},
    ]
    output = {}
    for name in schema.names:
        output[name] = json.dumps(conv, ensure_ascii=False) if name == "conversations" else row[name]
    return output


class PartWriter:
    def __init__(self, parts_dir, schema, part_size):
        self.parts_dir = Path(parts_dir)
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.part_size = part_size
        self.rows, self.indices = [], []
        existing = list(self.parts_dir.glob("part-*.parquet"))
        self.part_number = max([int(x.stem.split("-")[1]) for x in existing] + [-1]) + 1

    def add(self, index, row):
        self.indices.append(index)
        self.rows.append(row)
        if len(self.rows) >= self.part_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        stem = f"part-{self.part_number:06d}"
        final_parquet = self.parts_dir / f"{stem}.parquet"
        temp_parquet = self.parts_dir / f"{stem}.parquet.tmp"
        columns = {name: [row[name] for row in self.rows] for name in self.schema.names}
        pq.write_table(pa.Table.from_pydict(columns, schema=self.schema), temp_parquet)
        os.replace(temp_parquet, final_parquet)
        atomic_json(self.parts_dir / f"{stem}.indices.json", self.indices)
        log(f"checkpoint {stem}: {len(self.rows)} 条")
        self.part_number += 1
        self.rows, self.indices = [], []


async def run(args, indices):
    parquet = pq.ParquetFile(args.input)
    schema = parquet.schema_arrow
    completed = load_completed(args.parts_dir)
    selected = set(indices) - completed
    log(f"已完成 {len(completed):,}，本次待处理候选 {len(selected):,}")

    state_path = Path(args.work_dir) / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    target = int(state.get("target", args.min_target))
    counters = Counter(state.get("counters", {}))
    counters["kept"] = len(completed)
    if len(completed) >= target:
        log(f"已有 {len(completed):,} 条，达到目标 {target:,}，直接进入合并")
        state.update({"target": target, "counters": dict(counters)})
        return state
    writer = PartWriter(args.parts_dir, schema, args.part_size)
    task_queue = asyncio.Queue(maxsize=args.queue_size)
    result_queue = asyncio.Queue(maxsize=args.queue_size)
    stop = asyncio.Event()
    started = time.monotonic()
    deadline = started + args.deadline_hours * 3600
    clients = [AsyncOpenAI(api_key="EMPTY", base_url=url, timeout=args.timeout) for url in args.endpoints]
    worker_count = len(clients) * args.concurrency_per_endpoint

    async def producer():
        base = 0
        for batch in parquet.iter_batches(batch_size=2048):
            if stop.is_set():
                break
            data = batch.to_pydict()
            rows = batch.num_rows
            for offset in range(rows):
                if stop.is_set():
                    break
                index = base + offset
                if index not in selected:
                    continue
                parsed = parse_conversation(data["conversations"][offset])
                if parsed is None:
                    counters["late_filter"] += 1
                    continue
                row = {name: data[name][offset] for name in schema.names}
                await task_queue.put((index, row, parsed[0], parsed[1]))
            base += rows
        for _ in range(worker_count):
            await task_queue.put(None)

    async def worker(worker_id):
        client = clients[worker_id % len(clients)]
        while True:
            item = await task_queue.get()
            if item is None:
                break
            if stop.is_set():
                continue
            index, row, user, answer = item
            think, status = await call_teacher(client, args, user, answer)
            await result_queue.put((index, row, user, answer, think, status))
        await result_queue.put(None)

    async def collector():
        nonlocal target
        finished_workers = 0
        while finished_workers < worker_count:
            if time.monotonic() >= deadline:
                stop.set()
            item = await result_queue.get()
            if item is None:
                finished_workers += 1
                continue
            index, row, user, answer, think, status = item
            counters["attempted"] += 1
            counters[status] += 1
            if think:
                writer.add(index, build_output_row(row, user, answer, think, schema))
                counters["kept"] += 1

            elapsed = max(time.monotonic() - started, 1)
            if counters["kept"] >= args.benchmark_size and "benchmark_rate" not in state:
                rate = (counters["kept"] - len(completed)) / elapsed
                projected = int(len(completed) + rate * args.deadline_hours * 3600 * 0.80)
                target = max(args.min_target, min(args.max_target, projected))
                state.update({"benchmark_rate": rate, "target": target})
                log(f"基准完成: {rate:.2f} kept/s，动态目标 {target:,}")

            if counters["attempted"] % 100 == 0:
                rate = (counters["kept"] - len(completed)) / elapsed
                log(f"attempted={counters['attempted']:,} kept={counters['kept']:,}/{target:,} rate={rate:.2f}/s")
                state.update({"target": target, "counters": dict(counters), "elapsed_sec": elapsed})
                atomic_json(state_path, state)
            if counters["kept"] >= target:
                stop.set()
        writer.flush()
        state.update({"target": target, "counters": dict(counters), "finished_at": time.strftime("%F %T")})
        atomic_json(state_path, state)

    jobs = [asyncio.create_task(worker(i)) for i in range(worker_count)]
    producer_job = asyncio.create_task(producer())
    collector_job = asyncio.create_task(collector())
    await producer_job
    await asyncio.gather(*jobs)
    await collector_job
    return state


def merge_parts(parts_dir, output, schema):
    parts = sorted(Path(parts_dir).glob("part-*.parquet"))
    if not parts:
        raise RuntimeError("没有可合并的 parquet 分片")
    output = Path(output)
    temp = output.with_suffix(output.suffix + ".tmp")
    writer = pq.ParquetWriter(temp, schema)
    rows = 0
    try:
        for part in parts:
            table = pq.read_table(part)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        writer.close()
    check = pq.ParquetFile(temp).metadata.num_rows
    if check != rows:
        raise RuntimeError(f"合并校验失败: expected={rows}, actual={check}")
    os.replace(temp, output)
    log(f"合并完成: {output}，{rows:,} 行，{len(parts)} 个分片")
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--model", default="qwen2.5-vl-7b-instruct")
    parser.add_argument("--candidate-count", type=int, default=360000)
    parser.add_argument("--seed", default="minimind-cot-v1")
    parser.add_argument("--min-target", type=int, default=100000)
    parser.add_argument("--max-target", type=int, default=300000)
    parser.add_argument("--benchmark-size", type=int, default=1000)
    parser.add_argument("--deadline-hours", type=float, default=9.0)
    parser.add_argument("--concurrency-per-endpoint", type=int, default=6)
    parser.add_argument("--queue-size", type=int, default=1000)
    parser.add_argument("--part-size", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.work_dir = str(Path(args.work_dir))
    args.parts_dir = str(Path(args.work_dir) / "parts")
    return args


def main():
    args = parse_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    candidate_cache = Path(args.work_dir) / "candidates.json"
    indices = select_candidates(args.input, candidate_cache, args.candidate_count, args.seed)
    if args.prepare_only:
        return
    state = asyncio.run(run(args, indices))
    schema = pq.ParquetFile(args.input).schema_arrow
    rows = merge_parts(args.parts_dir, args.output, schema)
    state["merged_rows"] = rows
    atomic_json(Path(args.work_dir) / "state.json", state)


if __name__ == "__main__":
    main()
