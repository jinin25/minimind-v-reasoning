"""Deterministic generation evaluation for VQA/OCR/counting/short-answer tasks."""
import argparse, io, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_profiles import build_vlm_config
from model.model_vlm import MiniMindVLM
from trainer.trainer_utils import init_vlm_model

def norm(s): return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s.lower())
def units(s): return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", s.lower())
def f1(pred, ref):
    p, r = Counter(units(pred)), Counter(units(ref)); overlap = sum((p & r).values())
    return 0.0 if not overlap else 2 * overlap / (sum(p.values()) + sum(r.values()))
def answer_text(s):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", s, re.S | re.I)
    return m.group(1) if m else re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I).strip()
def canonical(s):
    s = answer_text(s)
    opt = re.search(r"(?:选项|option)?\s*[（(]?([a-zA-Z])[）)]", s)
    if opt: return "option:" + opt.group(1).lower()
    number = re.search(r"(?<!\w)-?\d+(?:\.\d+)?", s)
    if number and len(norm(s)) < 80: return "number:" + number.group(0)
    return norm(s)

p = argparse.ArgumentParser()
p.add_argument("--weight", required=True); p.add_argument("--data", default="dataset/general_generation_eval.parquet")
p.add_argument("--output", required=True); p.add_argument("--samples", type=int, default=-1)
p.add_argument("--max_new_tokens", type=int, default=128); p.add_argument("--reasoning", choices=["on", "off"], default="off")
p.add_argument("--device", default="cuda:0"); args = p.parse_args()
cfg = build_vlm_config("reason_vlm_109m", 1024, 0)
model, tok, processor = init_vlm_model(cfg, from_weight=args.weight, device=args.device, freeze_llm=2); model.eval()
t = pq.read_table(args.data); n = len(t) if args.samples < 0 else min(args.samples, len(t)); records=[]
for i in range(n):
    messages = json.loads(t["conversations"][i].as_py()); user = next(x["content"] for x in messages if x["role"] == "user")
    instruction = "\n请先在<think>中简短思考，再在<answer>中给出答案。" if args.reasoning == "on" else "\n只给出最终答案，不要展示思考过程。"
    content = user.replace("<image>", cfg.image_special_token * cfg.image_token_len) + instruction
    prompt = tok.apply_chat_template([{"role":"user","content":content}], tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=896).to(args.device)
    raw = t["image_bytes"][i].as_py(); raw = raw[0] if isinstance(raw, list) else raw
    pixels = {k:v.to(args.device) for k,v in MiniMindVLM.image2tensor(Image.open(io.BytesIO(raw)), processor).items()}
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=pixels,
                             do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    pred = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    ref = t["reference"][i].as_py() if "reference" in t.column_names else next(x["content"] for x in messages if x["role"] == "assistant")
    kind = t["category"][i].as_py() if "category" in t.column_names else "cot_reasoning"
    exact = canonical(pred) == canonical(ref); score=f1(answer_text(pred), answer_text(ref))
    records.append({"row":i,"category":kind,"reference":ref,"prediction":pred,"exact":exact,"token_f1":score,
                    "has_think":bool(re.search(r"<think>.*?</think>",pred,re.S)),"has_answer":bool(re.search(r"<answer>.*?</answer>",pred,re.S))})
    if (i+1)%20==0: print(f"generated={i+1}/{n}", flush=True)
groups=defaultdict(list)
for x in records: groups[x["category"]].append(x)
def summarize(xs): return {"samples":len(xs),"accuracy":sum(x["exact"] for x in xs)/len(xs),"token_f1":sum(x["token_f1"] for x in xs)/len(xs),"think_rate":sum(x["has_think"] for x in xs)/len(xs),"answer_format_rate":sum(x["has_answer"] for x in xs)/len(xs)}
result={"weight":args.weight,"reasoning":args.reasoning,"overall":summarize(records),"by_category":{k:summarize(v) for k,v in groups.items()},"records":records}
Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="records"},ensure_ascii=False,indent=2))
