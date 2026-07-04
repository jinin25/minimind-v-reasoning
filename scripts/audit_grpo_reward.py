"""Audit GRPO reward components on a fixed RL validation split."""
import argparse, json, os, sys
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_profiles import build_vlm_config
from trainer.trainer_utils import init_vlm_model
from trainer.train_grpo_vlm import RLInnovatorVLDataset, VLMGRPOCollator, LocalRewarder, _to_device_pixel_values

p=argparse.ArgumentParser(); p.add_argument('--weight',required=True); p.add_argument('--output',required=True)
p.add_argument('--data_dir',default='dataset/RL_Innovator-VL');p.add_argument('--samples',type=int,default=100)
p.add_argument('--device',default='cuda:0');p.add_argument('--max_gen_len',type=int,default=192);args=p.parse_args()
cfg=build_vlm_config('reason_vlm_109m',704,0)
model,tok,processor=init_vlm_model(cfg,from_weight=args.weight,device=args.device,freeze_llm=2);model.eval()
ds=RLInnovatorVLDataset(args.data_dir,tok,processor,split='val',val_ratio=.02,split_seed='minimind-v-grpo',
    max_prompt_len=512,image_special_token=cfg.image_special_token,image_token_len=cfg.image_token_len,max_samples=args.samples)
loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=VLMGRPOCollator(tok,512))
rewarder=LocalRewarder(format_weight=.3,tag_weight=.1,answer_weight=.6)
rows=[]
for i,b in enumerate(loader):
    ids=b['input_ids'].to(args.device); mask=b['attention_mask'].to(args.device); pixels=_to_device_pixel_values(b['pixel_values'],args.device)
    with torch.no_grad(),torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out=model.generate(ids,attention_mask=mask,pixel_values=pixels,max_new_tokens=args.max_gen_len,do_sample=False,
            eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
    completion='<think>\n'+tok.decode(out[0,ids.size(1):],skip_special_tokens=True)
    fmt=rewarder._format_reward(completion); tag=rewarder._tag_count_reward(completion)
    ans=rewarder._answer_reward(completion,b['answers'][0],b['answer_types'][0]); total=.3*fmt+.1*tag+.6*ans
    rows.append({'id':b['ids'][0],'answer_type':b['answer_types'][0],'reference':b['answers'][0],'completion':completion,
                 'format':fmt,'tag':tag,'answer':ans,'reward':total})
    if (i+1)%20==0: print(f'audited={i+1}/{len(ds)}',flush=True)
def mean(k): return sum(x[k] for x in rows)/len(rows)
result={'weight':args.weight,'samples':len(rows),'format_rate':mean('format'),'tag_score':mean('tag'),
        'answer_accuracy':mean('answer'),'mean_reward':mean('reward'),
        'reward_distribution':dict(Counter(str(x['reward']) for x in rows)),'rows':rows}
Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))
