"""Strictly curate 5K candidates and shorten samples whose answer/EOS is truncated."""
import argparse, hashlib, json, re
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

p=argparse.ArgumentParser(); p.add_argument('--input',default='dataset/short_answer_5k_distilled.parquet')
p.add_argument('--output',default='dataset/short_answer_5k_sft_v2.parquet'); p.add_argument('--max-length',type=int,default=768)
args=p.parse_args(); tok=AutoTokenizer.from_pretrained('model'); table=pq.read_table(args.input); changed=[]
targets={'multiple-choice':2000,'number':1500,'ocrtext':1500}; pools={k:[] for k in targets}; seen=set()
image_pad='<|image_pad|>'*64; close_ids=tok('</answer>',add_special_tokens=False).input_ids
eos_ids=tok(f'{tok.eos_token}\n',add_special_tokens=False).input_ids
def contains(values,pattern): return any(values[i:i+len(pattern)]==pattern for i in range(len(values)-len(pattern)+1))
def is_chinese(text): return len(re.findall(r'[\u3400-\u9fff]',text)) >= 8
for row in table.to_pylist():
    sid=str(row['id']); kind=str(row['answer_type']).lower()
    if sid in seen or kind not in targets: continue
    seen.add(sid); conv=json.loads(row['conversations'])
    target=next(x['content'] for x in conv if x['role']=='assistant')
    match=re.search(r'<answer>\s*(.*?)\s*</answer>',target,re.S|re.I); answer=match.group(1).strip() if match else ''
    valid=(kind=='multiple-choice' and bool(re.fullmatch(r'[A-D]',answer,re.I))) or \
          (kind=='number' and bool(re.fullmatch(r'-?(?:\d+(?:\.\d+)?|\.\d+)',answer.replace(',','')))) or \
          (kind=='ocrtext' and 0<len(answer)<=128)
    messages=[{'role':x['role'],'content':x['content'].replace('<image>',image_pad)} for x in conv]
    prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    ids=tok(prompt,add_special_tokens=False).input_ids[:args.max_length]
    valid = valid and contains(ids,close_ids) and contains(ids,eos_ids)
    if valid: pools[kind].append(row)
rows=[]
for kind,want in targets.items():
    ranked=sorted(pools[kind],key=lambda r:hashlib.sha256(f"warmup-v2:{r['id']}".encode()).hexdigest())
    if len(ranked)<want: raise RuntimeError(f'insufficient {kind}: {len(ranked)} < {want}')
    rows.extend(ranked[:want])
for row in rows:
    conv=json.loads(row['conversations'])
    messages=[{'role':x['role'],'content':x['content'].replace('<image>',image_pad)} for x in conv]
    prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    ids=tok(prompt,add_special_tokens=False).input_ids[:args.max_length]
    target=next(x for x in conv if x['role']=='assistant'); content=target['content']
    match=re.search(r'<think>\s*(.*?)\s*</think>',content,re.S|re.I)
    question=next(x['content'] for x in conv if x['role']=='user').replace('<image>','')
    if match and is_chinese(question) != is_chinese(match.group(1)):
        short='根据图像和问题中的信息可以确定答案。' if is_chinese(question) else 'The image and question provide the information needed.'
        target['content']=content[:match.start(1)]+short+content[match.end(1):]
        changed.append(str(row['id']))
    row['conversations']=json.dumps(conv,ensure_ascii=False)
pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),args.output,compression='zstd')
report={'input':args.input,'output':args.output,'rows':len(rows),'distribution':targets,
        'candidate_counts':{k:len(v) for k,v in pools.items()},'changed_count':len(changed),'changed_ids':changed}
Path(args.output+'.repair.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
