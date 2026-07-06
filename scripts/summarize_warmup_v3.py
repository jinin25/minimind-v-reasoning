import json
from pathlib import Path
r=Path('experiment_runs/warmup_v3');base=json.load(open(r/'cot_sft_rd02_diagnostics.json'));rows=[]
base_general=0.2273721514303659
for e in (1,2):
 d=json.load(open(r/f'warmup_v3_epoch{e}_diagnostics.json'));g=json.load(open(r/f'epoch{e}_general.json'))['overall']['token_f1'];s=d['summary']['normal']
 admitted=(s['multiple-choice']['accuracy']>=.20 and s['counting']['accuracy']>=.05 and d['visual_gain_vs_shuffle']>=.03 and g>=base_general*.9)
 rows.append({'epoch':e,'normal':s,'zero':d['summary']['zero'],'shuffle':d['summary']['shuffle'],'visual_gain_vs_shuffle':d['visual_gain_vs_shuffle'],'general_f1':g,'general_retention':g/base_general,'rft_admitted':admitted})
ok=[x for x in rows if x['rft_admitted']];out={'baseline':base['summary'],'epochs':rows,'rft_admitted':bool(ok),'recommended_epoch':max(ok,key=lambda x:x['normal']['macro_accuracy'])['epoch'] if ok else None}
(r/'admission_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
