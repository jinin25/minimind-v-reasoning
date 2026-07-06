"""Resumable multi-vLLM one-sentence reasoning distillation for short-answer SFT."""
import argparse, asyncio, base64, io, json, os, re, time
from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from openai import AsyncOpenAI

THINK=re.compile(r'<think>\s*(.*?)\s*</think>',re.S|re.I)
META_PATTERNS=re.compile(r'(?i)(reference answer|given answer|the answer follows|verified answer|参考答案|给定答案|答案是)')
def answer_str(x):
    if isinstance(x,list): return '' if not x else str(x[0]) if len(x)==1 else ', '.join(map(str,x))
    return '' if x is None else str(x)
def parse(text,question):
    m=THINK.search(text or ''); body=' '.join(m.group(1).split()).strip() if m else ''
    if not 8<=len(body)<=300 or '<answer' in body.lower() or META_PATTERNS.search(body): return None
    cjk=len(re.findall(r'[\u3400-\u9fff]',question)); body_cjk=len(re.findall(r'[\u3400-\u9fff]',body))
    if cjk>=2 and body_cjk<2:return None
    if cjk==0 and body_cjk>0:return None
    return body
def image_url(images):
    raw=next((x.get('bytes') for x in images or [] if isinstance(x,dict) and x.get('bytes')),None)
    if raw is None:return None
    im=Image.open(io.BytesIO(raw)).convert('RGB');buf=io.BytesIO();im.thumbnail((768,768));im.save(buf,format='JPEG',quality=85)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
def prompt(problem,answer,kind):
    return ("Study the image and question. The verified final answer is supplied only to keep your reasoning correct. "
      "Write exactly one concise, task-specific reasoning sentence inside <think>...</think>. Do not output the final answer, "
      "do not mention a reference/given answer, and use the question's language.\n\n"
      f"Task type: {kind}\nQuestion: {problem}\nVerified final answer: {answer}")
async def main(a):
    done=set();parts=Path(a.parts);parts.mkdir(parents=True,exist_ok=True)
    for f in parts.glob('part-*.ids.json'):done.update(json.loads(f.read_text()))
    clients=[AsyncOpenAI(api_key='EMPTY',base_url=x,timeout=120) for x in a.endpoints];sem=asyncio.Semaphore(a.concurrency*len(clients));rows=[];ids=[];counts=Counter();part=len(list(parts.glob('part-*.parquet')))
    async def one(rec,client):
        async with sem:
            ans=answer_str(rec['answer']);url=image_url(rec['images']); content=[{'type':'text','text':prompt(rec['problem'],ans,rec['answer_type'])}]
            if url:content.insert(0,{'type':'image_url','image_url':{'url':url}})
            for _ in range(3):
                try:
                    r=await client.chat.completions.create(model=a.model,messages=[{'role':'user','content':content}],temperature=.2,max_tokens=96)
                    think=parse(r.choices[0].message.content,rec['problem'])
                    if think:return rec,think
                except Exception: await asyncio.sleep(2)
            return rec,None
    def flush():
        nonlocal rows,ids,part
        if not rows:return
        pq.write_table(pa.table({'id':[x['id'] for x in rows],'answer_type':[x['answer_type'] for x in rows],'conversations':[x['conversations'] for x in rows],'image_bytes':[x['image_bytes'] for x in rows]}),parts/f'part-{part:05d}.parquet',compression='zstd')
        (parts/f'part-{part:05d}.ids.json').write_text(json.dumps(ids));part+=1;rows=[];ids=[]
    pf=pq.ParquetFile(a.input);pending=[];idx=0
    for b in pf.iter_batches(batch_size=128):
        d=b.to_pydict()
        for j in range(len(b)):
            rec={k:d[k][j] for k in d};sid=str(rec['id']);idx+=1
            if sid in done:continue
            pending.append(asyncio.create_task(one(rec,clients[(idx-1)%len(clients)])))
        if len(pending)>=a.queue:
            for fut in asyncio.as_completed(pending):
                rec,think=await fut
                if think:
                    ans=answer_str(rec['answer']);conv=[{'role':'user','content':rec['problem']},{'role':'assistant','content':f'<think>\n{think}\n</think>\n\n<answer>\n{ans}\n</answer>'}]
                    rows.append({'id':str(rec['id']),'answer_type':str(rec['answer_type']),'conversations':json.dumps(conv,ensure_ascii=False),'image_bytes':[x['bytes'] for x in rec['images'] if x.get('bytes')]});ids.append(str(rec['id']));counts['kept']+=1
                    if len(rows)>=a.part_size:flush()
                else:counts['failed']+=1
            pending=[];print(dict(counts),flush=True)
    for fut in asyncio.as_completed(pending):
        rec,think=await fut
        if think:
            ans=answer_str(rec['answer']);conv=[{'role':'user','content':rec['problem']},{'role':'assistant','content':f'<think>\n{think}\n</think>\n\n<answer>\n{ans}\n</answer>'}]
            rows.append({'id':str(rec['id']),'answer_type':str(rec['answer_type']),'conversations':json.dumps(conv,ensure_ascii=False),'image_bytes':[x['bytes'] for x in rec['images'] if x.get('bytes')]});ids.append(str(rec['id']));counts['kept']+=1
        else:counts['failed']+=1
    flush();tables=[pq.read_table(x) for x in sorted(parts.glob('part-*.parquet'))];pq.write_table(pa.concat_tables(tables),a.output,compression='zstd')
    Path(a.report).write_text(json.dumps({'counts':dict(counts),'rows':sum(len(x) for x in tables)},indent=2)+'\n');print(dict(counts))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='dataset/short_answer_5k_source.parquet');p.add_argument('--output',default='dataset/short_answer_5k_distilled.parquet');p.add_argument('--parts',default='dataset/short_answer_5k_parts');p.add_argument('--report',default='dataset/manifests/short_answer_5k_distill.json');p.add_argument('--model',default='qwen2.5-vl-7b-instruct');p.add_argument('--endpoints',nargs='+',default=['http://127.0.0.1:8000/v1','http://127.0.0.1:8011/v1','http://127.0.0.1:8002/v1','http://127.0.0.1:8003/v1']);p.add_argument('--concurrency',type=int,default=3);p.add_argument('--queue',type=int,default=48);p.add_argument('--part_size',type=int,default=250);asyncio.run(main(p.parse_args()))
