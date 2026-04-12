import argparse
import base64
import http.client
import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from statistics import mean

import pyarrow as pa
import pyarrow.parquet as pq


IMAGE_TOKEN_RE = re.compile(r"<\s*image[^>]*>", re.IGNORECASE)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(
	r"(?:^|\n)\s*(?:Answer|答案|最终答案)\s*[:：]\s*(.*)",
	re.IGNORECASE | re.DOTALL,
)

CAPTION_HINTS = (
	"描述这张图片",
	"描述图片",
	"请描述",
	"image caption",
	"describe this image",
	"caption",
)

BOILERPLATE_PREFIXES = (
	"这是图片",
	"从图中可以看出",
	"根据图片",
	"the image shows",
)

EXPLANATION_ANSWER_FOCUSED_RE = re.compile(
	r"参考答案|标准答案|上文答案|该答案|根据答案|答案中提到",
	re.IGNORECASE,
)
EXPLANATION_EVAL_STYLE_RE = re.compile(r"合理|正确|符合", re.IGNORECASE)


def percentile(values, p):
	if not values:
		return 0.0
	vals = sorted(values)
	if len(vals) == 1:
		return float(vals[0])
	rank = (len(vals) - 1) * (p / 100.0)
	low = int(rank)
	high = min(low + 1, len(vals) - 1)
	frac = rank - low
	return float(vals[low] * (1.0 - frac) + vals[high] * frac)


def safe_text(x):
	return x if isinstance(x, str) else ""


def has_question_mark(text):
	return "?" in text or "？" in text


def normalize_user(user_text):
	user = IMAGE_TOKEN_RE.sub("", safe_text(user_text))
	return "<image>\n" + user.strip()


def looks_like_caption(user_text):
	low = safe_text(user_text).strip().lower()
	if not low:
		return True
	if has_question_mark(low):
		return False
	return any(hint in low for hint in CAPTION_HINTS)


def normalize_for_similarity(text):
	text = safe_text(text).strip().lower()
	text = re.sub(r"\s+", "", text)
	text = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", text)
	return text


def similarity(a, b):
	na = normalize_for_similarity(a)
	nb = normalize_for_similarity(b)
	if not na or not nb:
		return 0.0
	return SequenceMatcher(None, na, nb).ratio()


def log(msg):
	print(msg, flush=True)


def parse_standard_conversation(conv_text):
	try:
		conv = json.loads(conv_text)
	except Exception:
		return None, None, None, "json_error"

	if not isinstance(conv, list) or len(conv) < 2:
		return None, None, None, "non_standard"

	first = conv[0] if isinstance(conv[0], dict) else {}
	second = conv[1] if isinstance(conv[1], dict) else {}
	user = safe_text(first.get("content"))
	assistant = safe_text(second.get("content"))

	if not user.strip() or not assistant.strip():
		return None, None, conv, "empty_text"

	return user, assistant, conv, "ok"


def get_single_image_bytes(image_obj):
	if image_obj is None:
		return None, "no_image_bytes"
	if isinstance(image_obj, (bytes, bytearray)):
		return bytes(image_obj), "single"
	if isinstance(image_obj, list):
		clean_items = [x for x in image_obj if isinstance(x, (bytes, bytearray)) and len(x) > 0]
		if not clean_items:
			return None, "no_image_bytes"
		if len(clean_items) == 1:
			return bytes(clean_items[0]), "single"
		return None, "multi"
	return None, "no_image_bytes"


def build_distill_prompt(question, original_answer):
	return (
		"<image>\n"
		f"问题：{question}\n\n"
		f"参考答案：{original_answer}\n\n"
		"请先基于图像与问题独立推理，再给出答案，并严格遵守以下约束：\n"
		"1) 解释只写 1-2 句话，且必须是“图像证据 -> 回答问题”的因果表达。\n"
		"2) 解释中禁止出现这些词：参考答案、标准答案、上文答案、该答案、根据答案、答案中提到。\n"
		"3) 解释中禁止出现“合理、正确、符合”等评价答案的措辞。\n"
		"4) 输出只能是两行：\n"
		"解释: <1-2句，面向问题推理>\n"
		"Answer: <参考答案原文>\n\n"
		"5) Answer 必须等于参考答案原文，不允许改写、扩写或缩写。"
	)


def is_answer_focused_explanation(explanation_text):
	txt = safe_text(explanation_text)
	if not txt:
		return False
	return bool(EXPLANATION_ANSWER_FOCUSED_RE.search(txt))


def has_eval_style_explanation(explanation_text):
	txt = safe_text(explanation_text)
	if not txt:
		return False
	return bool(EXPLANATION_EVAL_STYLE_RE.search(txt))


def _guess_mime(image_bytes):
	head = image_bytes[:16]
	if head.startswith(b"\x89PNG"):
		return "image/png"
	if head.startswith(b"\xff\xd8"):
		return "image/jpeg"
	if head[:6] in (b"GIF87a", b"GIF89a"):
		return "image/gif"
	if head.startswith(b"RIFF") and b"WEBP" in head:
		return "image/webp"
	return "image/jpeg"


def _http_error_detail(err):
	try:
		body = err.read().decode("utf-8", errors="ignore").strip()
		if len(body) > 300:
			body = body[:300] + "..."
		return body
	except Exception:
		return ""


def _truncate_for_log(text, max_chars):
	txt = safe_text(text)
	if max_chars <= 0:
		return txt
	if len(txt) <= max_chars:
		return txt
	return txt[:max_chars] + "..."


def is_answer_too_long(pred_answer, ref_answer, max_ratio):
	if max_ratio <= 0:
		return False
	pred = safe_text(pred_answer).strip()
	ref = safe_text(ref_answer).strip()
	if not pred or not ref:
		return False
	return len(pred) > max(1, int(len(ref) * max_ratio))


def _build_ssl_context(tls12_only=False, insecure=False):
	if insecure:
		ctx = ssl._create_unverified_context()
	else:
		ctx = ssl.create_default_context()
	if tls12_only and hasattr(ssl, "TLSVersion"):
		ctx.minimum_version = ssl.TLSVersion.TLSv1_2
		ctx.maximum_version = ssl.TLSVersion.TLSv1_2
	return ctx


def call_glm_distill(
	api_key,
	base_url,
	endpoint,
	model,
	prompt,
	image_bytes,
	timeout,
	max_retries,
	backoff_base,
	tls12_only=False,
	insecure=False,
):
	mime = _guess_mime(image_bytes)
	b64_img = base64.b64encode(image_bytes).decode("utf-8")
	image_url = f"data:{mime};base64,{b64_img}"

	payload = {
		"model": model,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": prompt},
					{"type": "image_url", "image_url": {"url": image_url}},
				],
			}
		],
		"stream": False,
	}

	if base_url.endswith("/"):
		base_url = base_url[:-1]
	if not endpoint.startswith("/"):
		endpoint = "/" + endpoint
	url = base_url + endpoint

	body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
	headers = {
		"Content-Type": "application/json",
		"Accept": "application/json",
		"Connection": "close",
		"User-Agent": "cot-distill/1.0",
		"Authorization": f"Bearer {api_key}",
	}

	last_error = "unknown"
	use_tls12 = tls12_only
	for attempt in range(max_retries + 1):
		req = urllib.request.Request(url, data=body, headers=headers, method="POST")
		try:
			ssl_ctx = _build_ssl_context(tls12_only=use_tls12, insecure=insecure)
			with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
				text = resp.read().decode("utf-8", errors="ignore")
			obj = json.loads(text)
			choices = obj.get("choices") or []
			if not choices:
				last_error = "empty_choices"
				raise ValueError(last_error)
			msg = choices[0].get("message") or {}
			content = msg.get("content")
			if isinstance(content, list):
				parts = []
				for item in content:
					if isinstance(item, dict) and item.get("type") == "text":
						parts.append(item.get("text", ""))
				content = "\n".join(parts)
			if not isinstance(content, str) or not content.strip():
				last_error = "empty_content"
				raise ValueError(last_error)
			return content.strip(), None
		except urllib.error.HTTPError as err:
			detail = _http_error_detail(err)
			last_error = f"HTTPError {err.code}: {detail or err.reason}"
			if attempt >= max_retries:
				break
			time.sleep(backoff_base * (2 ** attempt))
		except urllib.error.URLError as err:
			last_error = f"URLError: {err}"
			reason = getattr(err, "reason", None)
			if (
				not use_tls12
				and isinstance(reason, ssl.SSLError)
				and "unexpected eof while reading" in str(reason).lower()
			):
				use_tls12 = True
				log("[distill] ssl eof detected, retry with forced TLS1.2")
			if attempt >= max_retries:
				break
			time.sleep(backoff_base * (2 ** attempt))
		except ssl.SSLError as err:
			last_error = f"SSLError: {err}"
			if not use_tls12 and "unexpected eof while reading" in str(err).lower():
				use_tls12 = True
				log("[distill] ssl eof detected, retry with forced TLS1.2")
			if attempt >= max_retries:
				break
			time.sleep(backoff_base * (2 ** attempt))
		except (
			http.client.IncompleteRead,
			http.client.RemoteDisconnected,
			socket.timeout,
			TimeoutError,
			OSError,
			ValueError,
			json.JSONDecodeError,
		) as err:
			last_error = f"{type(err).__name__}: {err}"
			if attempt >= max_retries:
				break
			time.sleep(backoff_base * (2 ** attempt))

	return None, last_error


def parse_distilled_output(raw_text, original_answer):
	txt = safe_text(raw_text).strip()
	answer_match = ANSWER_RE.search(txt)

	think_match = THINK_RE.search(txt)
	if think_match:
		think_body = think_match.group(1).strip()
		think_lines = [ln.strip() for ln in think_body.splitlines() if ln.strip()]
	else:
		# 容错：若模型未加 think 标签，回退到 Answer 前文本并强制包装为 think。
		reasoning_prefix = txt if not answer_match else txt[: answer_match.start()]
		reasoning_prefix = reasoning_prefix.strip()
		reasoning_prefix = re.sub(
			r"^\s*(?:解释|原因分析|推理过程|思考过程|思路|分析过程)\s*[:：]\s*",
			"",
			reasoning_prefix,
			flags=re.IGNORECASE,
		)
		think_lines = [ln.strip() for ln in reasoning_prefix.splitlines() if ln.strip()]

	think_lines = [ln for ln in think_lines if not re.match(r"^(?:answer|答案)\s*[:：]", ln, flags=re.IGNORECASE)]
	if not think_lines:
		return None, None, "no_think"

	answer = safe_text(original_answer).strip()
	if not answer:
		return None, None, "no_answer"

	formatted = "<think>\n" + "\n".join(think_lines) + "\n</think>\n\n" + f"Answer: {answer}"
	return formatted, answer, None


def audit_dataset(parquet_path, batch_size, max_samples=None, log_every=0):
	pf = pq.ParquetFile(parquet_path)
	counters = Counter()
	user_lengths = []
	assistant_lengths = []

	seen = 0
	for batch in pf.iter_batches(batch_size=batch_size):
		data = batch.to_pydict()
		rows = len(next(iter(data.values()))) if data else 0

		for i in range(rows):
			if max_samples is not None and seen >= max_samples:
				break
			seen += 1
			counters["total"] += 1

			if log_every and seen % log_every == 0:
				log(f"[audit] processed={seen} parseable={counters['parseable']} single_image={counters['single_image']}")

			conv_raw = data.get("conversations", [None] * rows)[i]
			img_raw = data.get("image_bytes", [None] * rows)[i]

			if img_raw is None or (isinstance(img_raw, list) and len(img_raw) == 0):
				counters["anomaly_no_image_bytes"] += 1

			img_single, img_state = get_single_image_bytes(img_raw)
			if img_state == "single":
				counters["single_image"] += 1

			user, assistant, _, state = parse_standard_conversation(conv_raw)
			if state == "json_error" or state == "non_standard":
				counters["anomaly_non_standard_structure"] += 1
				continue
			if state == "empty_text":
				counters["anomaly_empty_text"] += 1
				continue

			counters["parseable"] += 1
			user_text = safe_text(user).strip()
			assistant_text = safe_text(assistant).strip()

			user_lengths.append(len(user_text))
			assistant_lengths.append(len(assistant_text))

			if IMAGE_TOKEN_RE.search(user_text):
				counters["user_contains_image_token"] += 1
			if has_question_mark(user_text):
				counters["user_is_question"] += 1

		if max_samples is not None and seen >= max_samples:
			break

	total = counters["total"] or 1
	parseable = counters["parseable"]
	parseable_base = parseable or 1

	audit = {
		"total_samples": counters["total"],
		"parseable_conversations_ratio": parseable / total,
		"single_image_ratio": counters["single_image"] / total,
		"user_length": {
			"mean": mean(user_lengths) if user_lengths else 0.0,
			"p50": percentile(user_lengths, 50),
			"p90": percentile(user_lengths, 90),
		},
		"assistant_length": {
			"mean": mean(assistant_lengths) if assistant_lengths else 0.0,
			"p50": percentile(assistant_lengths, 50),
			"p90": percentile(assistant_lengths, 90),
		},
		"user_contains_image_ratio": counters["user_contains_image_token"] / parseable_base,
		"user_question_ratio": counters["user_is_question"] / parseable_base,
		"anomalies": {
			"empty_text": counters["anomaly_empty_text"],
			"no_image_bytes": counters["anomaly_no_image_bytes"],
			"non_standard_structure": counters["anomaly_non_standard_structure"],
		},
	}
	return audit


def process_dataset(args, audit):
	api_key = os.getenv("DISTILL_API_KEY", "").strip()
	if not api_key:
		raise RuntimeError("缺少环境变量 DISTILL_API_KEY")

	pf = pq.ParquetFile(args.input)
	schema = pf.schema_arrow
	writer = pq.ParquetWriter(args.output, schema)

	stats = Counter()
	think_line_lengths = []
	similarity_scores = []
	api_latencies = []
	api_debug_fp = None

	if args.api_debug:
		dump_dir = os.path.dirname(args.api_debug_dump)
		if dump_dir:
			os.makedirs(dump_dir, exist_ok=True)
		api_debug_fp = open(args.api_debug_dump, "w", encoding="utf-8")
		log(f"[distill] api debug dump path={args.api_debug_dump}")

	log(
		f"[distill] start input={args.input} output={args.output} "
		f"model={args.model} max_samples={args.max_samples}"
	)

	processed = 0
	last_heartbeat = time.time()
	for batch in pf.iter_batches(batch_size=args.batch_size):
		data = batch.to_pydict()
		if not data:
			continue
		rows = len(next(iter(data.values())))

		out_cols = {name: [] for name in schema.names}

		for i in range(rows):
			if args.max_samples is not None and processed >= args.max_samples:
				break
			processed += 1
			stats["total"] += 1

			now = time.time()
			if args.heartbeat_sec > 0 and (now - last_heartbeat) >= args.heartbeat_sec:
				log(
					f"[distill] heartbeat processed={processed} keep={stats['keep']} "
					f"api_called={stats['api_called']} api_fail={stats['drop_api_fail']}"
				)
				last_heartbeat = now

			if args.log_every and processed % args.log_every == 0:
				current_keep_ratio = stats["keep"] / max(stats["total"], 1)
				log(
					f"[distill] processed={processed} keep={stats['keep']} "
					f"keep_ratio={current_keep_ratio:.3f} api_called={stats['api_called']} "
					f"api_fail={stats['drop_api_fail']}"
				)

			conv_raw = data.get("conversations", [None] * rows)[i]
			img_raw = data.get("image_bytes", [None] * rows)[i]

			user, assistant, _, state = parse_standard_conversation(conv_raw)
			if state in {"json_error", "non_standard"}:
				stats["drop_non_standard_structure"] += 1
				continue
			if state == "empty_text":
				stats["drop_empty_text"] += 1
				continue

			single_img, img_state = get_single_image_bytes(img_raw)
			if img_state == "no_image_bytes":
				stats["drop_no_image_bytes"] += 1
				continue
			if img_state == "multi":
				stats["drop_non_single_image"] += 1
				continue

			normalized_user = normalize_user(user)
			question_only = normalized_user.replace("<image>", "", 1).strip()

			if not (has_question_mark(question_only) or len(question_only) > args.min_user_len):
				stats["drop_not_question_or_too_short"] += 1
				continue
			if looks_like_caption(question_only):
				stats["drop_caption_like"] += 1
				continue

			prompt = build_distill_prompt(question_only, safe_text(assistant).strip())
			stats["api_called"] += 1
			if args.api_debug and stats["api_called"] <= args.api_debug_first_n:
				img_kb = len(single_img) / 1024.0
				log(
					f"[distill] api_call sample={processed} api_idx={stats['api_called']} "
					f"img_kb={img_kb:.1f} q_len={len(question_only)}"
				)
			api_t0 = time.time()
			distilled_raw, api_err = call_glm_distill(
				api_key=api_key,
				base_url=args.api_base_url,
				endpoint=args.api_endpoint,
				model=args.model,
				prompt=prompt,
				image_bytes=single_img,
				timeout=args.timeout,
				max_retries=args.max_retries,
				backoff_base=args.backoff_base,
				tls12_only=args.tls12_only,
				insecure=args.api_insecure_skip_verify,
			)
			lat = time.time() - api_t0
			api_latencies.append(lat)
			if args.api_debug and stats["api_called"] <= args.api_debug_first_n:
				log(
					f"[distill] api_done sample={processed} api_idx={stats['api_called']} "
					f"latency={lat:.2f}s ok={bool(distilled_raw)}"
				)

			if args.api_debug and api_debug_fp is not None:
				debug_record = {
					"sample": processed,
					"api_idx": stats["api_called"],
					"ok": bool(distilled_raw),
					"latency_sec": round(lat, 4),
					"api_error": api_err,
					"question": question_only,
					"ref_answer": safe_text(assistant).strip(),
					"prompt": prompt,
					"glm_raw": safe_text(distilled_raw),
				}
				api_debug_fp.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
				api_debug_fp.flush()

				if stats["api_called"] <= args.api_debug_first_n:
					preview = safe_text(distilled_raw)
					if not preview:
						preview = safe_text(api_err)
					preview = _truncate_for_log(preview, args.api_debug_print_chars)
					log(
						f"[distill] api_raw sample={processed} api_idx={stats['api_called']}\n"
						f"{preview}"
					)

			if not distilled_raw:
				stats["drop_api_fail"] += 1
				if args.log_every and stats["drop_api_fail"] <= 3:
					log(f"[distill] api_fail sample={processed} err={api_err}")
				continue

			formatted_assistant, pred_answer, parse_err = parse_distilled_output(
				distilled_raw, assistant
			)
			if parse_err == "no_think":
				stats["drop_no_think"] += 1
				continue
			if parse_err == "no_answer":
				stats["drop_no_answer"] += 1
				continue

			ref_answer = safe_text(assistant).strip()
			if is_answer_too_long(pred_answer, ref_answer, args.max_answer_ratio):
				stats["drop_answer_too_long"] += 1
				continue

			think_block = THINK_RE.search(formatted_assistant)
			think_lines = []
			if think_block:
				think_lines = [ln.strip() for ln in think_block.group(1).splitlines() if ln.strip()]

			explanation_text = " ".join(think_lines)
			if is_answer_focused_explanation(explanation_text):
				stats["drop_answer_focused_explanation"] += 1
				continue
			if has_eval_style_explanation(explanation_text):
				stats["drop_explanation_eval_style"] += 1
				continue

			if len(formatted_assistant) < args.min_total_len or len(think_lines) < args.min_think_lines:
				stats["drop_too_short"] += 1
				continue

			sim = similarity(pred_answer, ref_answer)
			similarity_scores.append(sim)
			if sim < args.similarity_threshold:
				stats["drop_inconsistent_answer"] += 1
				continue

			think_line_lengths.append(len(think_lines))

			new_conv = [
				{"role": "user", "content": normalized_user},
				{"role": "assistant", "content": formatted_assistant},
			]
			new_conv_raw = json.dumps(new_conv, ensure_ascii=False)

			for name in schema.names:
				if name == "conversations":
					out_cols[name].append(new_conv_raw)
				else:
					out_cols[name].append(data[name][i])

			stats["keep"] += 1

		keep_rows = len(out_cols["conversations"]) if "conversations" in out_cols else 0
		if keep_rows > 0:
			out_table = pa.Table.from_pydict(out_cols, schema=schema)
			writer.write_table(out_table)
			if args.log_every:
				log(f"[distill] wrote batch rows={keep_rows}")

		if args.max_samples is not None and processed >= args.max_samples:
			break

	writer.close()
	if api_debug_fp is not None:
		api_debug_fp.close()

	total = stats["total"] or 1
	distill_report = {
		"total": stats["total"],
		"keep": stats["keep"],
		"keep_ratio": stats["keep"] / total,
		"api_called": stats["api_called"],
		"answer_focused_explanation_rate": (
			stats["drop_answer_focused_explanation"] / max(stats["api_called"], 1)
		),
		"avg_api_latency_sec": mean(api_latencies) if api_latencies else 0.0,
		"avg_think_lines": mean(think_line_lengths) if think_line_lengths else 0.0,
		"answer_consistency_rate": (
			(stats["keep"] / max(stats["keep"] + stats["drop_inconsistent_answer"], 1))
		),
		"avg_answer_similarity": mean(similarity_scores) if similarity_scores else 0.0,
		"drops": {
			"no_think": stats["drop_no_think"],
			"no_answer": stats["drop_no_answer"],
			"too_short": stats["drop_too_short"],
			"answer_too_long": stats["drop_answer_too_long"],
			"answer_focused_explanation": stats["drop_answer_focused_explanation"],
			"explanation_eval_style": stats["drop_explanation_eval_style"],
			"inconsistent_answer": stats["drop_inconsistent_answer"],
			"api_fail": stats["drop_api_fail"],
			"empty_text": stats["drop_empty_text"],
			"no_image_bytes": stats["drop_no_image_bytes"],
			"non_standard_structure": stats["drop_non_standard_structure"],
			"non_single_image": stats["drop_non_single_image"],
			"not_question_or_too_short": stats["drop_not_question_or_too_short"],
			"caption_like": stats["drop_caption_like"],
		},
	}

	log(
		f"[distill] done total={distill_report['total']} keep={distill_report['keep']} "
		f"keep_ratio={distill_report['keep_ratio']:.3f} api_called={distill_report['api_called']}"
	)

	return distill_report


def parse_args():
	parser = argparse.ArgumentParser(description="SFT parquet CoT distillation pipeline")
	parser.add_argument("--input", default="sft_i2t.parquet", help="输入 parquet 路径")
	parser.add_argument("--output", default="sft_i2t_cot_distilled.parquet", help="输出 parquet 路径")
	parser.add_argument("--report", default="distill_report.json", help="统计报告 JSON 路径")
	parser.add_argument("--audit_only", action="store_true", help="仅执行审计")
	parser.add_argument("--batch_size", type=int, default=512)
	parser.add_argument("--max_samples", type=int, default=None, help="限制处理样本数，便于小规模试跑")

	parser.add_argument("--api_base_url", default="https://open.bigmodel.cn/api/paas/v4")
	parser.add_argument("--api_endpoint", default="/chat/completions")
	parser.add_argument("--model", default="glm-4.6v")
	parser.add_argument("--timeout", type=int, default=60)
	parser.add_argument("--max_retries", type=int, default=3)
	parser.add_argument("--backoff_base", type=float, default=1.0)
	parser.add_argument("--tls12_only", action="store_true", help="强制仅使用 TLS1.2（用于排查 SSL EOF）")
	parser.add_argument(
		"--api_insecure_skip_verify",
		action="store_true",
		help="跳过 HTTPS 证书校验，仅用于网络排查，不建议生产使用",
	)

	parser.add_argument("--min_user_len", type=int, default=15)
	parser.add_argument("--min_total_len", type=int, default=40)
	parser.add_argument("--min_think_lines", type=int, default=1)
	parser.add_argument("--similarity_threshold", type=float, default=0.85)
	parser.add_argument(
		"--max_answer_ratio",
		type=float,
		default=1.20,
		help="蒸馏答案相对参考答案的最大长度倍率，超过则丢弃",
	)
	parser.add_argument("--log_every", type=int, default=100, help="每处理多少条打印一次进度，0 表示关闭")
	parser.add_argument("--heartbeat_sec", type=int, default=15, help="心跳日志间隔秒数，0 表示关闭")
	parser.add_argument("--api_debug", action="store_true", help="开启 API 调用调试日志")
	parser.add_argument("--api_debug_first_n", type=int, default=10, help="仅打印前 N 次 API 调用调试日志")
	parser.add_argument(
		"--api_debug_dump",
		default="glm_api_debug.jsonl",
		help="保存每次 API 调用输入输出到 JSONL（配合 --api_debug 使用）",
	)
	parser.add_argument(
		"--api_debug_print_chars",
		type=int,
		default=1200,
		help="控制台打印原始返回的最大字符数，0 表示不截断",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	log("[main] phase0 audit start")
	audit = audit_dataset(args.input, args.batch_size, args.max_samples, args.log_every)
	log("[main] phase0 audit done")
	result = {"audit": audit, "config": vars(args)}

	if not args.audit_only:
		log("[main] phase1-4 distill start")
		distill = process_dataset(args, audit)
		result["distill"] = distill
		log("[main] phase1-4 distill done")

	with open(args.report, "w", encoding="utf-8") as f:
		json.dump(result, f, ensure_ascii=False, indent=2)

	print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
