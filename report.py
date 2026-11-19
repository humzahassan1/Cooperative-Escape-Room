"""Generate pilot tables and figures from saved experiment artifacts."""
import json
import platform
import importlib.metadata as metadata
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from learning import evaluate

def main():
    root=Path(__file__).parent; runs=root/'runs'; results=[]
    for label,folder,filename in [('PPO + communication','communication','evaluation.json'),('PPO no communication','no_communication','evaluation.json'),('PPO messages silenced','communication','evaluation_silence.json'),('PPO random messages','communication','evaluation_random.json')]:
        data=json.loads((runs/folder/filename).read_text()); data['label']=label; results.append(data)
    for controller in ['random','scripted']:
        data=evaluate(controller=controller); data['label']=controller.title()
        (runs/f'{controller}.json').write_text(json.dumps(data,indent=2)); results.append(data)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for label,folder in [('Communication','communication'),('No communication','no_communication')]:
        rows=[r for r in json.loads((runs/folder/'training.json').read_text()) if 'success_rate' in r]
        axes[0].plot([r['steps'] for r in rows],[100*r['success_rate'] for r in rows],marker='o',label=label,alpha=.8)
    axes[0].set(xlabel='Joint environment steps',ylabel='Success (%)',title='Pilot learning curves (40 episodes/check)',ylim=(-1,100)); axes[0].legend()
    axes[1].barh([r['label'] for r in results],[100*r['success_rate'] for r in results],color=['#348fca','#f1a340','#8a71c2','#ba87c9','#8f9ba5','#35ac8d'])
    axes[1].set(xlabel='Success (%)',title='Final evaluation (200 episodes)',xlim=(0,105)); axes[1].invert_yaxis()
    fig.savefig(root/'results.png',dpi=170); plt.close(fig)
    lines=['# Pilot results','','One training seed (7), fixed room and starts, balanced hidden clue. Each PPO condition trained for 501,760 joint environment steps on CPU. Evaluation uses stochastic actions with learning disabled and seeds 10000–10199. No claim of a reliable communication advantage is supported.','','| Controller | Success | Successful steps | Outcomes |','|---|---:|---:|---|']
    for r in results:
        counts=dict(Counter(x['outcome'] for x in r['records']))
        length='N/A' if r['successful_episode_length'] is None else f"{r['successful_episode_length']:.1f}"
        lines.append(f"| {r['label']} | {r['success_rate']:.1%} | {length} | {counts} |")
    lines+=['','## Interpretation','','The scripted agents solve the task, but both final PPO policies fail in this budget. This establishes a working experimental pipeline, not learned cooperation. Sparse coordinated rewards and simultaneous policy learning are plausible difficulties; this pilot does not isolate the cause. Message ablations cannot demonstrate meaningful communication when the original policy already fails.','','The full grid is publicly observable except the clue. Movement and timing can also convey information. The scripted reference uses an explicitly programmed symbol convention and is not a learned policy. Fixed layouts, one training seed, memoryless policies, and only two clue values limit generalization.','','## Runtime','']
    for r in results[:2]:lines.append(f"- {r['label']}: {r['training_seconds']:.1f} seconds (includes periodic and final normal evaluation; excludes message ablations).")
    lines+=['','## Next experiment','','Preserve this pilot. Test a separately labeled curriculum (unlocking first, then exiting) or potential-based distance shaping with matched rewards across conditions. Repeat across at least three training seeds before making comparative claims. Do not quietly replace these failed results.','','![Pilot figure](results.png)']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    versions={'python':platform.python_version(),'platform':platform.platform()}
    for name in ['torch','numpy','pygame','pygame-ce','matplotlib']:
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:pass
    (root/'tested_environment.json').write_text(json.dumps(versions,indent=2))
    print('\n'.join(lines[:12]))

if __name__=='__main__':main()
