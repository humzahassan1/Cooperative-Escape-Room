"""Paired clue intervention: same starting positions, policy weights and action-sampling noise."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from environment import EscapeRoom, IDS
from learning import Policy, load


@torch.no_grad()
def paired_test(policies, communication, pairs=100, seed=30000):
    choices = []
    outcomes = []
    for clue in (0, 1):
        torch.manual_seed(seed)
        rooms = [EscapeRoom(communication) for _ in range(pairs)]
        obs = [room.reset(seed+i, clue=clue)[0] for i,room in enumerate(rooms)]
        choice = [None] * pairs
        for step in range(100):
            actions = []
            for agent, policy in zip(IDS,policies):
                batch = torch.tensor(np.stack([o[agent] for o in obs]))
                actions.append(policy.sample(batch)[0].tolist())
            for i,room in enumerate(rooms):
                if room.done:
                    continue
                was_locked = not room.unlocked
                obs[i], *_ = room.step({agent: actions[j][i] for j,agent in enumerate(IDS)})
                if was_locked and (room.unlocked or room.outcome == 'wrong switch'):
                    choice[i] = room.config['switches'].index(list(room.pos[1]))
        choices.append(choice)
        outcomes.append([r.outcome == 'success' for r in rooms])
    return dict(pairs=pairs, seed=seed,
                correct_switch_both_clues=sum(a==0 and b==1 for a,b in zip(*choices)),
                escape_both_clues=sum(a and b for a,b in zip(*outcomes)),
                escapes=[sum(x) for x in outcomes])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    args=parser.parse_args()
    policies,data=load(args.checkpoint)
    torch.manual_seed(data['config']['seed'])
    initial=[Policy(),Policy()]
    results={label:paired_test(pair,data['config']['communication']) for label,pair in
             [('initial',initial),('trained',policies)]}
    args.checkpoint.with_name('paired_clues.json').write_text(json.dumps(results,indent=2))
    print(results)
