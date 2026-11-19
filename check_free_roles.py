"""Replay saved learners in the actual game and measure transfer to new layouts."""
import json
from pathlib import Path
from collections import Counter
from environment import EscapeRoom, random_room
from free_roles import Policy, trials


def evaluate():
    folder = Path('runs/free_roles')
    layouts = [random_room(seed) for seed in range(18000,18025)]
    records = []
    for path in sorted(folder.glob('seed*.json')):
        data = json.loads(path.read_text())
        policy = Policy(data['genome'])
        fixed = trials(data['genome'])
        assert fixed == data['enabled']
        varied = Counter()
        for config in layouts:
            env = EscapeRoom(config=dict(config,interchangeable=True))
            for clue in (0,1):
                for holder in (0,1):
                    env.clue_holder = holder
                    obs, _ = env.reset(0,clue)
                    while not env.done:
                        obs, *_ = env.step(policy.act(obs))
                    varied[env.outcome] += 1
        row = dict(seed=data['seed'],generations=data['generations'],
                   fixed=sum(r['outcome']=='success' for r in fixed),
                   silenced=sum(r['outcome']=='success' for r in data['silenced']),
                   plate_holders=dict(Counter(r['plate_agent'] for r in fixed if r['outcome']=='success')),
                   new_layout_outcomes=dict(varied))
        records.append(row)
        print(row)
    (folder/'summary.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    evaluate()
