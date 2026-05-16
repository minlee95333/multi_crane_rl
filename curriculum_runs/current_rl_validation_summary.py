import json, math
from pathlib import Path
p=Path(__file__).resolve().with_name('current_rl_validation_result.json')
r=json.loads(p.read_text())
def pct(a,b): return (a-b)/a*100
seen=r['seen']; unseen=r['unseen']
rows=[]
for split, data in [('seen',seen),('unseen',unseen)]:
    m=data['MAPPO-trained']; before=data['RL-before']; near=data['Nearest']; rand=data['Random']
    rows.append({
        'split':split,
        'mappo_makespan':m['makespan'],
        'before_improve_pct':pct(before['makespan'],m['makespan']),
        'nearest_improve_pct':pct(near['makespan'],m['makespan']),
        'random_improve_pct':pct(rand['makespan'],m['makespan']),
        'completion':m['completeRate'],
        'soft':m['soft'],
        'hardMask':m['hardMask'],
        'travel':m['travel'],
        'setup':m['setup'],
        'move':m['move'],
    })
print(json.dumps({'train_best':r['trainSummary']['best'], 'elapsedSec':r['elapsedSec'], 'modelStats':r['trainSummary']['modelStats'], 'comparisons':rows}, ensure_ascii=False, indent=2))
