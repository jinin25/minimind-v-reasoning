"""Build/use a fixed per-task RL manifest and evaluate greedy plus pass@4."""
import argparse, glob, hashlib, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from bisect import bisect_right
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_profiles import build_vlm_config
from trainer.trainer_utils import init_vlm_model
from trainer.train_grpo_vlm import RLInnovatorVLDataset, VLMGRPOCollator, LocalRewarder, _to_device_pixel_values

TASK_MAP={'multiple-choice':'multiple-choice','number':'counting','ocrtext':'ocr','bbox':'grounding'}

def build_manifest(data_dir,path,n=200):
    selected={v:[] for v in TASK_MAP.values()}
    for shard in sorted(glob.glob(os.path.join(data_dir,'RL_part*.parquet'))):
        pf=pq.ParquetFile(shard);base=0
        for b in pf.iter_batches(batch_size=4096,columns=['id','answer_type']):
            for j,(sid,kind) in enumerate(zip(b['id'].to_pylist(),b['answer_type'].to_pylist())):
                task=TASK_MAP.get(str(kind).lower())
                if task:
                    rank=hashlib.sha256(f'short-answer-eval-v1:{sid}'.encode()).hexdigest()
                    selected[task].append((rank,shard,base+j,str(sid),str(kind).lower()))
            base+=len(b)
    rows=[]
    for task,items in selected.items():
        for rank,shard,row,sid,kind in sorted(items)[:n]: rows.append({'task':task,'path':shard,'row':row,'id':sid,'answer_type':kind,'rank':rank})
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    return rows

def summarize(rows):
    out={}
    for task in TASK_MAP.values():
        xs=[x for x in rows if x['task']==task]
        out[task]={'samples':len(xs),'greedy_accuracy':None if task=='grounding' else sum(x['greedy_correct'] for x in xs)/len(xs),
                   'pass_at_4':None if task=='grounding' else sum(x['pass_at_4'] for x in xs)/len(xs),
                   'format_rate':sum(x['format'] for x in xs)/len(xs),'parseable_rate':sum(x['parseable'] for x in xs)/len(xs),
                   'repeated_answer_close_rate':sum(x['repeated_answer_close'] for x in xs)/len(xs)}
    valid=[v for k,v in out.items() if k!='grounding']
    out['macro']={'greedy_accuracy':sum(x['greedy_accuracy'] for x in valid)/len(valid),'pass_at_4':sum(x['pass_at_4'] for x in valid)/len(valid)}
    return out

p=argparse.ArgumentParser();p.add_argument('--weight',required=True);p.add_argument('--output',required=True)
p.add_argument('--data_dir',default='dataset/RL_Innovator-VL');p.add_argument('--manifest',default='dataset/manifests/short_answer_eval_v1.jsonl')
p.add_argument('--samples_per_task',type=int,default=200);p.add_argument('--device',default='cuda:0');p.add_argument('--max_gen_len',type=int,default=96);args=p.parse_args()
manifest=list(map(json.loads,Path(args.manifest).read_text().splitlines())) if Path(args.manifest).exists() else build_manifest(args.data_dir,args.manifest,args.samples_per_task)
cfg=build_vlm_config('reason_vlm_109m',608,0);model,tok,processor=init_vlm_model(cfg,from_weight=args.weight,device=args.device,freeze_llm=2);model.eval()
answer_stop=tok('</answer>',add_special_tokens=False).input_ids
ds=RLInnovatorVLDataset(args.data_dir,tok,processor,split='all',max_prompt_len=512,image_special_token=cfg.image_special_token,image_token_len=cfg.image_token_len,max_samples=1)
ds.samples=[(x['path'],x['row'],x['id']) for x in manifest]; collate=VLMGRPOCollator(tok,512);rewarder=LocalRewarder()
results=[]
loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=2,collate_fn=collate)
offset=0
for b in loader:
    metas=manifest[offset:offset+len(b['ids'])];offset+=len(metas)
    ids=b['input_ids'].to(args.device);mask=b['attention_mask'].to(args.device);pixels=_to_device_pixel_values(b['pixel_values'],args.device)
    generated=[]
    for k in range(1 if metas[0]['task']=='grounding' else 5):
        with torch.no_grad(),torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out=model.generate(ids,attention_mask=mask,pixel_values=pixels,max_new_tokens=args.max_gen_len,do_sample=k>0,
                temperature=.8,top_p=.9,top_k=50,repetition_penalty=1.05,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
                stop_token_sequences=[answer_stop])
        generated.append(['<think>\n'+text for text in tok.batch_decode(out[:,ids.size(1):],skip_special_tokens=True)])
    for j,meta in enumerate(metas):
        completions=[group[j] for group in generated];greedy=completions[0]
        scores=[rewarder._answer_reward(x,b['answers'][j],b['answer_types'][j]) for x in completions]
        answer_match=re.search(r'<answer>\s*(.*?)\s*</answer>',greedy,re.S|re.I)
        results.append({'id':meta['id'],'task':meta['task'],'answer_type':b['answer_types'][j],'reference':b['answers'][j],
            'greedy':greedy,'greedy_correct':scores[0],'pass_at_4':max(scores[1:]) if len(scores)>1 else 0,
            'format':rewarder._format_reward(greedy),'parseable':float(bool(answer_match and answer_match.group(1).strip())),
            'repeated_answer_close':float(greedy.lower().count('</answer>') > 1)})
    if len(results)%20==0: print(f'evaluated={len(results)}/{len(ds)}',flush=True)
payload={'weight':args.weight,'manifest':args.manifest,'summary':summarize(results),'rows':results}
Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(payload['summary'],ensure_ascii=False,indent=2))
