"""Verify canonical answers receive full local reward on fixed RL examples."""
import argparse, json, os, sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trainer.train_grpo_vlm import LocalRewarder, _safe_answer_to_str

p=argparse.ArgumentParser();p.add_argument('--data',default='dataset/RL_Innovator-VL/RL_part000000.parquet')
p.add_argument('--samples_per_type',type=int,default=200);p.add_argument('--output',default='experiment_runs/short_answer_diagnostic/oracle.json');args=p.parse_args()
aliases={'multiple-choice':'multiple-choice','number':'number','ocrtext':'ocrtext'}
rows={k:[] for k in aliases}
pf=pq.ParquetFile(args.data)
for b in pf.iter_batches(batch_size=1024,columns=['id','answer','answer_type']):
    for sid,answer,kind in zip(b['id'].to_pylist(),b['answer'].to_pylist(),b['answer_type'].to_pylist()):
        kind=str(kind).lower()
        if kind in rows and len(rows[kind])<args.samples_per_type: rows[kind].append((str(sid),answer))
    if all(len(v)>=args.samples_per_type for v in rows.values()): break
r=LocalRewarder(format_weight=.3,tag_weight=.1,answer_weight=.6);details=[]
for kind,items in rows.items():
    for sid,answer in items:
        canonical=_safe_answer_to_str(answer); completion=f'<think>oracle</think><answer>{canonical}</answer>'
        fmt=r._format_reward(completion);tag=r._tag_count_reward(completion);ans=r._answer_reward(completion,answer,kind);total=.3*fmt+.1*tag+.6*ans
        details.append({'id':sid,'type':kind,'format':fmt,'tag':tag,'answer':ans,'total':total})
summary={'samples':len(details),'by_type':{},'all_passed':all(x['format']==x['tag']==x['answer']==x['total']==1 for x in details)}
for kind in rows:
    xs=[x for x in details if x['type']==kind];summary['by_type'][kind]={'samples':len(xs),'passed':sum(x['total']==1 for x in xs)}
Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps({'summary':summary,'details':details},indent=2)+'\n')
print(json.dumps(summary,indent=2));raise SystemExit(0 if summary['all_passed'] else 1)
