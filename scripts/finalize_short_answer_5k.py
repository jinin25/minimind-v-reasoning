"""Validate distilled rows and select an exact balanced 5K training set."""
import hashlib, json, re
from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
TARGET={'multiple-choice':2000,'number':1500,'ocrtext':1500}
src=pq.read_table('dataset/short_answer_5k_distilled.parquet'); chosen=[];seen=set();counts=Counter();fail=Counter()
for i in range(len(src)):
    sid=str(src['id'][i].as_py());kind=str(src['answer_type'][i].as_py()).lower();conv=src['conversations'][i].as_py();images=src['image_bytes'][i].as_py()
    if kind not in TARGET or counts[kind]>=TARGET[kind]:continue
    if sid in seen:fail['duplicate']+=1;continue
    try:m=json.loads(conv);answer=m[-1]['content']
    except Exception:fail['json']+=1;continue
    if not re.fullmatch(r'<think>\s*.+?\s*</think>\s*<answer>\s*.+?\s*</answer>',answer,re.S):fail['xml']+=1;continue
    if not images or any(not x for x in images):fail['image']+=1;continue
    if re.search(r'(?i)(reference answer|given answer|verified answer|参考答案|给定答案)',answer):fail['meta']+=1;continue
    seen.add(sid);counts[kind]+=1;chosen.append(i)
if any(counts[k]<TARGET[k] for k in TARGET):raise RuntimeError(f'insufficient clean rows: {counts}, failures={fail}')
out=src.take(pa.array(chosen,type=pa.int64()));pq.write_table(out,'dataset/short_answer_5k_sft.parquet',compression='zstd')
sha=hashlib.sha256(Path('dataset/short_answer_5k_sft.parquet').read_bytes()).hexdigest();report={'rows':len(out),'by_type':dict(counts),'failures':dict(fail),'sha256':sha}
Path('dataset/manifests/short_answer_5k_final.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
