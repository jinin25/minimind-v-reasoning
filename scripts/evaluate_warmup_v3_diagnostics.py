"""Evaluate normal, zero-image and shuffled-image generation on fixed task sets."""
import argparse,io,json,os,re,sys
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from model.model_profiles import build_vlm_config
from model.model_vlm import MiniMindVLM
from trainer.trainer_utils import init_vlm_model
from trainer.train_grpo_vlm import LocalRewarder
p=argparse.ArgumentParser();p.add_argument('--weight',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--samples-per-task',type=int,default=0);args=p.parse_args()
t=pq.read_table('dataset/warmup_v3_diagnostic_eval.parquet')
if args.samples_per_task:
 chosen=[];counts=defaultdict(int)
 for i,task in enumerate(t['task'].to_pylist()):
  if counts[task]<args.samples_per_task:chosen.append(i);counts[task]+=1
 t=t.take(pa.array(chosen,type=pa.int64()))
cfg=build_vlm_config('reason_vlm_109m',768,0)
model,tok,processor=init_vlm_model(cfg,from_weight=args.weight,device=args.device,freeze_llm=2);model.eval();rewarder=LocalRewarder()
stop=tok('</answer>',add_special_tokens=False).input_ids; rows=[]; n=len(t)
images=[]
for i in range(n):
 raw=t['image_bytes'][i].as_py();raw=raw[0] if isinstance(raw,list) else raw
 images.append(MiniMindVLM.image2tensor(Image.open(io.BytesIO(raw)),processor))
for mode in ('normal','zero','shuffle'):
 for i in range(n):
  q=t['question'][i].as_py();task=t['task'][i].as_py();ref=t['reference'][i].as_py()
  q=q.replace('<image>','').strip(); prompt=cfg.image_special_token*cfg.image_token_len+'\n'+q+'\nRespond with <think>brief reasoning</think><answer>final answer</answer>.'
  text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
  inp=tok(text,return_tensors='pt',truncation=True,max_length=608).to(args.device)
  src=i if mode!='shuffle' else (i+1)%n; pix={k:v.to(args.device) for k,v in images[src].items()}
  if mode=='zero': pix={k:torch.zeros_like(v) for k,v in pix.items()}
  with torch.no_grad(),torch.cuda.amp.autocast(dtype=torch.bfloat16):
   out=model.generate(inp.input_ids,attention_mask=inp.attention_mask,pixel_values=pix,do_sample=False,max_new_tokens=96,
       eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,stop_token_sequences=[stop])
  pred='<think>\n'+tok.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True)
  typ={'multiple-choice':'multiple-choice','counting':'number','ocr':'ocrtext'}[task]
  score=rewarder._answer_reward(pred,[str(ref)],typ);fmt=rewarder._format_reward(pred)
  rows.append({'id':t['id'][i].as_py(),'task':task,'mode':mode,'reference':ref,'prediction':pred,'correct':score,'format':fmt})
  if (i+1)%50==0: print(mode,i+1,'/',n,flush=True)
summary={}
for mode in ('normal','zero','shuffle'):
 summary[mode]={}
 for task in ('multiple-choice','counting','ocr'):
  xs=[x for x in rows if x['mode']==mode and x['task']==task]
  summary[mode][task]={'accuracy':sum(x['correct'] for x in xs)/len(xs),'format_rate':sum(x['format'] for x in xs)/len(xs)}
 valid=[summary[mode][x]['accuracy'] for x in ('multiple-choice','counting','ocr')]
 summary[mode]['macro_accuracy']=sum(valid)/len(valid)
payload={'weight':args.weight,'summary':summary,'visual_gain_vs_shuffle':summary['normal']['macro_accuracy']-summary['shuffle']['macro_accuracy'],'rows':rows}
Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');print(json.dumps(payload['summary'],indent=2))
