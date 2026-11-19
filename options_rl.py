"""Independent policy gradients over navigation goals and a learned symbol channel."""
import argparse
import json
from collections import deque
from functools import lru_cache
from pathlib import Path
import numpy as np
from environment import EscapeRoom, IDS, MOVES, random_room


def navigate(observation, agent, goal, walls):
    # Goals index the four public objects: plate, switch 1, switch 2, exit.
    points = np.rint(observation[:12].reshape(6, 2) * [8, 6]).astype(int)
    start = tuple(points[agent])
    target = tuple(points[2 + goal])
    if observation[13 + agent]:
        return 4
    if start == target:
        return 5 if goal in (1, 2) else 4
    blocked = set(map(tuple, walls))
    if not observation[12]:
        blocked.add(tuple(points[5]))
    if not observation[14 - agent]:
        blocked.add(tuple(points[1 - agent]))
    queue = deque([(start, None)])
    seen = {start}
    while queue:
        (x, y), first = queue.popleft()
        for action, (dx, dy) in enumerate(MOVES[:4]):
            cell = (x + dx, y + dy)
            if not (0 < cell[0] < 8 and 0 < cell[1] < 6) or cell in blocked or cell in seen:
                continue
            move = action if first is None else first
            if cell == target:
                return move
            seen.add(cell)
            queue.append((cell, move))
    return 4


def play(clue, before, after, config=None):
    room = EscapeRoom(config=config)
    obs, _ = room.reset(0, clue=clue)
    obs, reward, *_ = room.step({k:(4,0) for k in IDS})
    total = reward['A']
    while not room.done:
        goals = after if room.unlocked else before
        actions = {k:(navigate(obs[k], i, goals[i], room.config.get('walls', ())), 0)
                   for i,k in enumerate(IDS)}
        obs, reward, *_ = room.step(actions)
        total += reward['A']
        # With fixed goals, unchanged positions and door state imply a timeout.
        if not room.done and all(a in (4,5) for a,_ in actions.values()) and not room.unlocked:
            total -= .01 * (100-room.steps)
            break
    return total, room.outcome == 'success'


def probabilities(logits):
    weights = np.exp(logits - logits.max(axis=-1,keepdims=True))
    return weights / weights.sum(axis=-1,keepdims=True)


def train(seed, channel):
    rng = np.random.default_rng(seed)
    # Separate tables for A's signal, A/B's locked goals, then A/B's unlocked goals.
    tables = [rng.normal(0,.08,shape) for shape in [(2,2),(2,4),(3,4),(1,4),(1,4)]]
    baseline = 0.
    for update in range(500):
        n = 128
        clue = rng.integers(2,size=n)
        samples = []
        def draw(index, states):
            prob = probabilities(tables[index][states])
            action = (rng.random((n,1)) > prob.cumsum(axis=1)).sum(axis=1).clip(0,prob.shape[1]-1)
            samples.append((states, action, prob))
            return action
        symbol = draw(0,clue) + 1
        first_a = draw(1,clue)
        first_b = draw(2,symbol if channel else np.zeros(n,int))
        last_a = draw(3,np.zeros(n,int))
        last_b = draw(4,np.zeros(n,int))
        returns = np.array([cached_return(*key) for key in zip(clue,first_a,first_b,last_a,last_b)])
        for table,(states,action,prob) in zip(tables,samples):
            gradient = -prob
            gradient[np.arange(n),action] += 1
            np.add.at(table,states,.45*(returns-baseline)[:,None]*gradient/n)
        baseline = .9*baseline + .1*returns.mean()
    return [t.tolist() for t in tables]


@lru_cache(maxsize=2048)
def cached_return(clue,a,b,c,d):
    return play(clue,(a,b),(c,d))[0]


class Controller:
    def __init__(self, tables, channel=True):
        self.tables = list(map(np.asarray,tables))
        self.channel = channel

    def act(self, obs, walls):
        a,b = obs['A'],obs['B']
        clue = int(a[16:18].argmax())
        symbol = int(self.tables[0][clue].argmax()) + 1
        received = int(b[-3:].argmax()) if self.channel else 0
        if a[15] == 1:
            return {'A':(4,symbol),'B':(4,0)}
        if a[12]:
            goals = [self.tables[3][0].argmax(),self.tables[4][0].argmax()]
        else:
            goals = [self.tables[1][clue].argmax(),self.tables[2][received].argmax()]
        return {k:(navigate(obs[k],i,goals[i],walls),symbol if i==0 else 0) for i,k in enumerate(IDS)}


def evaluate(tables, rooms, mode):
    ctrl = Controller(tables,mode!='silenced')
    counts = []
    for config in rooms:
        pair = []
        for clue in (0,1):
            env = EscapeRoom(config=config)
            obs,_ = env.reset(0,clue=clue)
            while not env.done:
                actions = ctrl.act(obs,config.get('walls',()))
                if mode=='swapped':
                    action,symbol=actions['A']
                    actions['A']=(action,3-symbol)
                obs,*_=env.step(actions)
            pair.append(env.outcome=='success')
        counts.append(pair)
    return dict(escapes=int(np.sum(counts)),episodes=2*len(rooms),both_clues=sum(all(x) for x in counts))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('runs/options'))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rooms=[random_room(s) for s in range(17000,17100)]
    results=[]
    for seed in (7,8,9):
        for channel in (True,False):
            tables=train(seed,channel)
            name=f'seed{seed}_{"comm" if channel else "silent"}'
            (args.out/f'{name}.json').write_text(json.dumps(dict(tables=tables,channel=channel)))
            conditions=['normal','swapped','silenced'] if channel else ['silenced']
            row=dict(seed=seed,channel=channel,tests={k:evaluate(tables,rooms,k) for k in conditions})
            results.append(row)
            print(row,flush=True)
    (args.out/'results.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
