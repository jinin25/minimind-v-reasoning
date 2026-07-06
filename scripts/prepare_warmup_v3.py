"""Build leakage-free diagnostics and a 70/30 short-answer/general replay mixture."""
import glob, hashlib, heapq, json, re
from collections import defaultdict
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'dataset'; M=D/'manifests'; M.mkdir(exist_ok=True)
dist=pq.read_table(D/'short_answer_5k_distilled.parquet').to_pylist()
train=pq.read_table(D/'short_answer_5k_sft_v2.parquet').to_pylist(); train_ids={str(x['id']) for x in train}
source=pq.read_table(D/'short_answer_5k_source.parquet').to_pylist()
def rank(seed,sid): return hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest()
def answer_of(conv):
    m=re.search(r'<answer>\s*(.*?)\s*</answer>',json.loads(conv)[1]['content'],re.S|re.I);return m.group(1).strip() if m else ''
def raw_image(row):
    raw=row['image_bytes'] if 'image_bytes' in row else [x['bytes'] for x in row['images']]
    return raw if isinstance(raw,list) else [raw]

# Same-source held-out sets: unused distilled PixMo/Infographic candidates.
eval_rows=[]
for kind,task in [('number','counting'),('ocrtext','ocr')]:
    pool=[x for x in dist if x['answer_type']==kind and str(x['id']) not in train_ids]
    for x in sorted(pool,key=lambda z:rank('v3-eval',str(z['id'])))[:200]:
        conv=json.loads(x['conversations']); q=conv[0]['content']; a=answer_of(x['conversations'])
        eval_rows.append({'id':str(x['id']),'task':task,'question':q,'reference':a,'image_bytes':raw_image(x)})

# Balanced strict four-choice MC from the full RL pool, disjoint from SFT.
by=defaultdict(list)
for path in sorted(glob.glob(str(D/'RL_Innovator-VL'/'RL_part*.parquet'))):
    pf=pq.ParquetFile(path);base=0
    for batch in pf.iter_batches(batch_size=4096,columns=['id','answer','answer_type']):
        for j,x in enumerate(batch.to_pylist()):
            sid=str(x['id']);a=str(x['answer'][0]).strip().upper() if x.get('answer') else ''
            if x['answer_type']=='multiple-choice' and sid not in train_ids and a in 'ABCD':
                by[a].append((rank('v3-mc-eval',sid),path,base+j,sid))
        base+=len(batch)
def read_parquet_row(path,index):
    pf=pq.ParquetFile(path);offset=0
    for rg in range(pf.num_row_groups):
        n=pf.metadata.row_group(rg).num_rows
        if index<offset+n:return pf.read_row_group(rg).slice(index-offset,1).to_pylist()[0]
        offset+=n
    raise IndexError(index)
for label in 'ABCD':
    pool=sorted(by[label])
    if len(pool)<50: raise RuntimeError(f'insufficient held-out MC {label}: {len(pool)}')
    for _,path,index,sid in pool[:50]:
        x=read_parquet_row(path,index);eval_rows.append({'id':sid,'task':'multiple-choice','question':x['problem'],
                                          'reference':label,'image_bytes':[z['bytes'] for z in x['images']]})
pq.write_table(pa.Table.from_pylist(eval_rows),D/'warmup_v3_diagnostic_eval.parquet',compression='zstd')

# 2K label-balanced MC + 3K counting. Oversampling is explicit and deterministic.
short_by=defaultdict(list)
for x in train:
    if x['answer_type']=='multiple-choice': short_by[answer_of(x['conversations']).upper()].append(x)
count=[x for x in train if x['answer_type']=='number']
short=[]
for label in 'ABCD':
    pool=sorted(short_by[label],key=lambda z:rank('v3-short',str(z['id'])))
    short.extend(pool[i%len(pool)] for i in range(500))
count=sorted(count,key=lambda z:rank('v3-count',str(z['id'])))
short.extend(count[i%len(count)] for i in range(3000))

# Deterministic 10K General replay, excluding the fixed General validation conversations.
eval_hash=set()
for x in pq.read_table(D/'general_generation_eval.parquet',columns=['conversations']).to_pylist():
    eval_hash.add(hashlib.sha256(x['conversations'].encode()).hexdigest())
best=[]
for batch in pq.ParquetFile(D/'sft_i2t.parquet').iter_batches(batch_size=8192):
    for x in batch.to_pylist():
        h=hashlib.sha256(x['conversations'].encode()).hexdigest()
        if h in eval_hash: continue
        key=int(h,16)
        item=(-key,h,x)
        if len(best)<10000: heapq.heappush(best,item)
        elif key < -best[0][0]: heapq.heapreplace(best,item)
general=[x for _,_,x in sorted(best,key=lambda z:z[1])]

# Materialize weighted epoch mixture: 23,334 short + 10,000 General = 70/30.
mixed=[]
for i in range(23334):
    x=short[i%len(short)]; mixed.append({'id':f'short:{i}:{x["id"]}','answer_type':x['answer_type'],
        'conversations':x['conversations'],'image_bytes':raw_image(x)[0]})
for i,x in enumerate(general): mixed.append({'id':f'general:{i}','answer_type':'general',
    'conversations':x['conversations'],'image_bytes':raw_image(x)[0]})
mixed=sorted(mixed,key=lambda x:rank('v3-mix',x['id']))
pq.write_table(pa.Table.from_pylist(mixed),D/'warmup_v3_mixed.parquet',compression='zstd')

# Unique MC pool for rejection sampling.
rft=[]
for x in train:
    if x['answer_type']=='multiple-choice' and answer_of(x['conversations']).upper() in 'ABCD': rft.append(x)
rft=sorted(rft,key=lambda x:rank('v3-rft',str(x['id'])))[:1000]
pq.write_table(pa.Table.from_pylist(rft),D/'warmup_v3_rft_pool.parquet',compression='zstd')
manifest={'diagnostic_rows':len(eval_rows),'diagnostic_by_task':{'multiple-choice':200,'counting':200,'ocr':200},
          'short_unique_base':5000,'materialized_short':23334,'general_replay':10000,'mixture_rows':len(mixed),
          'rft_pool':len(rft),'train_eval_overlap':len(train_ids & {x['id'] for x in eval_rows})}
(M/'warmup_v3.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
