"""Build a deterministic multi-source 5.8K buffer for 5K short-answer distillation."""
import glob, hashlib, io, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image

OUT='dataset/short_answer_5k_source.parquet'; REPORT='dataset/manifests/short_answer_5k_v1.json'
EVAL='dataset/manifests/short_answer_eval_v1.jsonl'; TARGET={'multiple-choice':2200,'number':1800,'ocrtext':1800}
excluded={json.loads(x)['id'] for x in Path(EVAL).read_text().splitlines()}
def rank(s):return hashlib.sha256(f'short-answer-5k-v2:{s}'.encode()).hexdigest()
def choose(items,n):return sorted(items,key=lambda x:rank(x['id']))[:n]
records=[]

# Multiple-choice: verified RL samples with embedded image bytes.
mc=[]
for path in sorted(glob.glob('dataset/Innovator-VL-RL-172K/RL_part*.parquet')):
    pf=pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=512,columns=['id','images','problem','answer','answer_type','problem_type','source','prompt_type']):
        for row in pa.Table.from_batches([b]).to_pylist():
            sid=str(row['id'])
            if str(row['answer_type']).lower()=='multiple-choice' and sid not in excluded and row['images'] and row['answer']:
                mc.append({'id':sid,'images':row['images'],'problem':row['problem'],'answer':row['answer'],'answer_type':'multiple-choice','problem_type':'stem','source':'Innovator-VL-RL-172K','prompt_type':'normal'})
records.extend(choose(mc,TARGET['multiple-choice']))

# OCR: short InfographicVQA answers with embedded image bytes.
ocr=[]
for path in sorted(glob.glob('dataset/short_answer_sources/DocVQA/InfographicVQA/train-*.parquet')):
    for b in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['questionId','question','answers','image']):
        for row in pa.Table.from_batches([b]).to_pylist():
            answers=[str(x).strip() for x in (row['answers'] or []) if str(x).strip()]
            raw=(row['image'] or {}).get('bytes')
            if answers and raw and len(answers[0])<=64:
                sid='infovqa:'+str(row['questionId']);ocr.append({'id':sid,'images':[{'bytes':raw,'path':(row['image'] or {}).get('path')}],
                    'problem':'<image>\n'+str(row['question']).strip(),'answer':[answers[0]],'answer_type':'ocrtext','problem_type':'ocr',
                    'source':'lmms-lab/DocVQA/InfographicVQA','prompt_type':'normal'})
records.extend(choose(ocr,TARGET['ocrtext']))

# Counting: deterministically choose URL metadata, then download and validate images concurrently.
pix=[]
for path in sorted(glob.glob('dataset/short_answer_sources/pixmo-count/data/train-*.parquet')):
    for b in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for row in pa.Table.from_batches([b]).to_pylist():
            sid='pixmo-count:'+str(row['image_sha256']);pix.append({'id':sid,'url':row['image_url'],'sha':row['image_sha256'],'count':row['count'],'label':row['label']})
pix=choose(pix,2200)
def download(x):
    try:
        r=requests.get(x['url'],timeout=30);r.raise_for_status();raw=r.content
        image=Image.open(io.BytesIO(raw));image.verify()
        if image.width<16 or image.height<16:return None
        return {'id':x['id'],'images':[{'bytes':raw,'path':None}],
            'problem':f"<image>\nHow many {x['label']} are in the image? Answer with only the number.",
            'answer':[str(x['count'])],'answer_type':'number','problem_type':'counting','source':'allenai/pixmo-count','prompt_type':'normal'}
    except Exception:return None
with ThreadPoolExecutor(max_workers=32) as pool:
    downloaded=[x for x in (f.result() for f in as_completed([pool.submit(download,x) for x in pix])) if x]
downloaded=choose(downloaded,TARGET['number']);records.extend(downloaded)

counts={k:sum(x['answer_type']==k for x in records) for k in TARGET}
if any(counts[k]<TARGET[k] for k in TARGET):raise RuntimeError(f'insufficient valid samples: {counts}, required={TARGET}')
records=sorted(records,key=lambda x:rank(x['id']))
pq.write_table(pa.Table.from_pylist(records),OUT,compression='zstd')
sha=hashlib.sha256(Path(OUT).read_bytes()).hexdigest();report={'version':'short-answer-5k-v2','output':OUT,'buffer_rows':len(records),'buffer_by_type':counts,
 'final_targets':{'multiple-choice':2000,'number':1500,'ocrtext':1500},'excluded_fixed_eval_ids':len(excluded),'sources':{
 'multiple-choice':'InnovatorLab/Innovator-VL-RL-172K','number':'allenai/pixmo-count','ocrtext':'lmms-lab/DocVQA/InfographicVQA'},'sha256':sha}
Path(REPORT).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
