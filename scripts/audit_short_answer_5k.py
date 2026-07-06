"""Audit the fixed 5K short-answer SFT dataset and render deterministic previews."""
import argparse, hashlib, io, json, re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def extract_block(text, tag):
    match = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S | re.I)
    return match.group(1).strip() if match else ""


def has_cjk(text):
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    return cjk >= 8


def option_letters(question):
    return set(re.findall(r"(?:^|\n)\s*([A-D])(?:\s*[.)、:]|\s*$)", question, re.M | re.I))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset/short_answer_5k_sft.parquet')
    p.add_argument('--output-dir', default='experiment_runs/short_answer_warmup_v2/audit')
    p.add_argument('--tokenizer', default='model')
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(args.data)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    expected = {'multiple-choice': 2000, 'number': 1500, 'ocrtext': 1500}
    counts, failures, previews = Counter(), defaultdict(list), defaultdict(list)
    ids = []
    image_pad = '<|image_pad|>' * 64
    eos_ids = tok(f'{tok.eos_token}\n', add_special_tokens=False).input_ids

    for i in range(len(table)):
        sid = str(table['id'][i].as_py()); kind = str(table['answer_type'][i].as_py()).lower()
        ids.append(sid); counts[kind] += 1
        try:
            conv = json.loads(table['conversations'][i].as_py())
            question = next(x['content'] for x in conv if x['role'] == 'user')
            target = next(x['content'] for x in conv if x['role'] == 'assistant')
            think, answer = extract_block(target, 'think'), extract_block(target, 'answer')
        except Exception as exc:
            failures['schema'].append({'id': sid, 'error': str(exc)}); continue
        try:
            raw = table['image_bytes'][i].as_py(); raw = raw[0] if isinstance(raw, list) else raw
            Image.open(io.BytesIO(raw)).verify()
        except Exception as exc:
            failures['image'].append({'id': sid, 'error': str(exc)}); continue
        if not think or not answer:
            failures['xml_or_empty'].append({'id': sid})
        if kind == 'multiple-choice':
            letters = option_letters(question)
            if len(letters) < 4:
                failures['mc_options_image_only'].append({'id': sid, 'answer': answer})
            elif not re.fullmatch(r'[A-D]', answer, re.I) or answer.upper() not in {x.upper() for x in letters}:
                failures['mc_answer_not_in_options'].append({'id': sid, 'answer': answer, 'options': sorted(letters)})
        elif kind == 'number' and not re.fullmatch(r'-?(?:\d+(?:\.\d+)?|\.\d+)', answer.replace(',', '')):
            failures['invalid_number'].append({'id': sid, 'answer': answer})
        elif kind == 'ocrtext' and (not answer or len(answer) > 128):
            failures['invalid_ocr'].append({'id': sid, 'answer': answer})
        q_no_image = question.replace('<image>', '').strip()
        if has_cjk(q_no_image) != has_cjk(think):
            failures['language_mismatch'].append({'id': sid, 'question': q_no_image[:120], 'think': think[:120]})

        messages = [{'role': x['role'], 'content': x['content'].replace('<image>', image_pad)} for x in conv]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = tok(prompt, add_special_tokens=False).input_ids[:768]
        close_ids = tok('</answer>', add_special_tokens=False).input_ids
        def contains(seq):
            return any(token_ids[j:j+len(seq)] == seq for j in range(len(token_ids)-len(seq)+1))
        if not contains(close_ids) or not contains(eos_ids):
            failures['truncated_answer_or_eos'].append({'id': sid, 'tokens_before_truncation': len(tok(prompt, add_special_tokens=False).input_ids)})
        if len(previews[kind]) < 30:
            previews[kind].append((sid, raw, q_no_image, answer))

    duplicates = [sid for sid, n in Counter(ids).items() if n > 1]
    if duplicates: failures['duplicate_id'] = duplicates
    if dict(counts) != expected: failures['distribution'] = [{'actual': dict(counts), 'expected': expected}]

    for kind, rows in previews.items():
        sheet = Image.new('RGB', (1500, 6 * 230), 'white'); draw = ImageDraw.Draw(sheet)
        for n, (sid, raw, question, answer) in enumerate(rows):
            x, y = (n % 5) * 300, (n // 5) * 230
            image = Image.open(io.BytesIO(raw)).convert('RGB'); image.thumbnail((280, 160))
            sheet.paste(image, (x + 10, y + 5))
            caption = f'{sid[:20]}\nQ: {question[:55]}\nA: {answer[:35]}'
            draw.multiline_text((x + 10, y + 168), caption, fill='black', spacing=2)
        sheet.save(out / f'preview_{kind}.jpg', quality=90)

    hard_fail_keys = {'schema','image','xml_or_empty','mc_answer_not_in_options','invalid_number',
                      'invalid_ocr','language_mismatch','truncated_answer_or_eos','duplicate_id','distribution'}
    passed = not any(failures.get(k) for k in hard_fail_keys)
    payload = {'rows': len(table), 'distribution': dict(counts), 'unique_ids': len(set(ids)),
               'sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
               'passed': passed, 'failure_counts': {k: len(v) for k,v in failures.items()},
               'failures': dict(failures)}
    (out / 'audit.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k != 'failures'}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__': main()
