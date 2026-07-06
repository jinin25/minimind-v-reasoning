"""Create the GRPO admission report from all Warmup-v2 epoch evaluations."""
import argparse, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--run-dir', default='experiment_runs/short_answer_warmup_v2')
p.add_argument('--baseline-general', default='experiment_runs/cot_eval_final/rd02_general_off.json'); args=p.parse_args()
root=Path(args.run_dir); baseline=json.loads(Path(args.baseline_general).read_text())
base_f1=baseline.get('overall',{}).get('token_f1', 0.2273721514)
rows=[]
for epoch in range(1,4):
    rl=json.loads((root/f'epoch{epoch}/rl_tasks.json').read_text()); general=json.loads((root/f'epoch{epoch}/general.json').read_text())
    s=rl['summary']; macro=s['macro']['greedy_accuracy']; gf1=general['overall']['token_f1']
    repeat=max(s[t].get('repeated_answer_close_rate',0) for t in ('multiple-choice','counting','ocr'))
    passed=(s['multiple-choice']['greedy_accuracy']>=.20 and s['counting']['greedy_accuracy']>=.05 and
            s['ocr']['greedy_accuracy']>=.05 and macro>=.10 and macro-0.013333333333333334>=.05 and
            gf1>=base_f1*.90 and repeat<.05)
    weight=Path(f'out/short_answer_warmup_v2_epoch{epoch}_768.pth')
    rows.append({'epoch':epoch,'multiple_choice':s['multiple-choice']['greedy_accuracy'],
                 'counting':s['counting']['greedy_accuracy'],'ocr':s['ocr']['greedy_accuracy'],
                 'macro':macro,'pass_at_4':s['macro']['pass_at_4'],'general_f1':gf1,
                 'general_retention':gf1/base_f1 if base_f1 else None,'max_repeat_rate':repeat,
                 'weight':str(weight),'sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'gate_passed':passed})
eligible=[x for x in rows if x['gate_passed']]
chosen=max(eligible,key=lambda x:(x['macro'],x['general_f1'])) if eligible else None
report={'baseline_general_f1':base_f1,'epochs':rows,'grpo_admitted':bool(chosen),
        'recommended_checkpoint':chosen['weight'] if chosen else None,
        'next_action':'run 100-200 sample GRPO probe' if chosen else 'stop GRPO and inspect task/data failures'}
(root/'admission_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
