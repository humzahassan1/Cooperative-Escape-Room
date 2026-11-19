"""Deterministic joint-step simulation. No rendering or learning dependencies."""
import json
from pathlib import Path
import numpy as np

IDS = ('A', 'B')
MOVES = ((0,-1),(0,1),(-1,0),(1,0),(0,0),(0,0))
OBS_VERSION = 1

class EscapeRoom:
    def __init__(self, communication=True, config=None, message_count=3):
        self.config = config or json.loads(Path(__file__).with_name('room.json').read_text())
        self.communication = communication
        self.message_count = int(message_count)
        self.reset(0)

    def reset(self, seed=None, clue=None):
        self.rng = np.random.default_rng(seed)
        self.clue = int(self.rng.integers(0,2)) if clue is None else int(clue)
        self.pos = [tuple(p) for p in self.config['starts']]
        self.escaped = [False,False]
        self.messages = [0,0]
        self.unlocked = False
        self.steps = 0
        self.done = False
        self.outcome = 'running'
        return self.observations(), self.info()

    def info(self):
        return dict(outcome=self.outcome, steps=self.steps, unlocked=self.unlocked)

    def observations(self):
        c = self.config
        scale = np.array([c['width']-1,c['height']-1])
        public = list(np.array(self.pos).reshape(-1) / np.tile(scale,2))
        for p in [c['plate'], *c['switches'], c['exit']]:
            public.extend(np.array(p)/scale)
        public.extend([float(self.unlocked),*map(float,self.escaped),1-self.steps/c['max_steps']])
        result = {}
        for i,k in enumerate(IDS):
            clue = [float(self.clue==j) if i==getattr(self,'clue_holder',0) else 0. for j in range(2)]
            message = [float(self.messages[1-i]==j) for j in range(self.message_count)]
            result[k] = np.array(public+clue+message,dtype=np.float32)
        return result

    def step(self, actions):
        if self.done:
            raise RuntimeError('reset required after episode end')
        pairs = [tuple(map(int,actions[k])) for k in IDS]
        if any(a not in range(6) or m not in range(self.message_count) for a,m in pairs):
            raise ValueError(f'actions must be (movement 0..5, message 0..{self.message_count - 1})')
        c = self.config
        old = self.pos.copy()
        proposed = old.copy()
        for i,(a,m) in enumerate(pairs):
            if self.escaped[i]: continue
            dx,dy = MOVES[a]
            x,y = old[i][0]+dx,old[i][1]+dy
            if 0<x<c['width']-1 and 0<y<c['height']-1 and [x,y] not in c.get('walls', ()) and (self.unlocked or (x,y)!=tuple(c['exit'])):
                proposed[i]=(x,y)
        if not any(self.escaped):
            if proposed[0]==proposed[1] or (proposed[0]==old[1] and proposed[1]==old[0]):
                proposed=old.copy()
        self.pos=proposed
        self.messages=[m if self.communication else 0 for a,m in pairs]
        self.steps+=1
        reward=-.01
        terminated=False
        holders = (0,1) if c.get('interchangeable', False) else (0,)
        for holder in holders:
            partner = 1-holder
            if self.unlocked or self.pos[holder]!=tuple(c['plate']) or pairs[partner][0]!=5:
                continue
            for j,p in enumerate(c['switches']):
                if self.pos[partner]==tuple(p):
                    if j==self.clue:
                        self.unlocked=True
                        reward+=1
                    else:
                        terminated=True
                        self.outcome='wrong switch'
                        reward-=1
        if self.unlocked:
            self.escaped=[e or p==tuple(c['exit']) for e,p in zip(self.escaped,self.pos)]
            if all(self.escaped):
                terminated=True
                self.outcome='success'
                reward+=5
        truncated=not terminated and self.steps>=c['max_steps']
        if truncated: self.outcome='timeout'
        self.done=terminated or truncated
        return self.observations(), {k:reward for k in IDS}, terminated, truncated, self.info()

def scripted(obs, walls=()):
    """Decentralized reference: B reads only its received message, never A's clue."""
    from collections import deque
    actions={}
    for i,k in enumerate(IDS):
        o=obs[k]; coords=np.rint(o[:12]*np.tile([8,6],6)).astype(int).reshape(6,2)
        positions=coords[:2]; plate,s0,s1,exit_pos=coords[2:]
        message=int(np.argmax(o[-3:]))
        clue=int(np.argmax(o[16:18]))+1 if i==0 else message
        target=exit_pos if o[12] else (plate if i==0 else [s0,s1][max(1,clue)-1])
        start=tuple(positions[i]); goal=tuple(target)
        action=4
        if not o[13+i] and (i==0 or o[12] or message):
            if start==goal:
                action=5 if i==1 and not o[12] else 4
            else:
                blocked=set(map(tuple, walls))
                if not o[13+1-i]: blocked.add(tuple(positions[1-i]))
                q=deque([(start,None)]); seen={start}
                while q:
                    p,first=q.popleft()
                    if p==goal:
                        action=first; break
                    for a,(dx,dy) in enumerate(MOVES[:4]):
                        n=(p[0]+dx,p[1]+dy)
                        if 0<n[0]<8 and 0<n[1]<6 and n not in blocked and n not in seen and (o[12] or n!=tuple(exit_pos)):
                            seen.add(n); q.append((n,a if first is None else first))
        actions[k]=(action,clue if i==0 else 0)
    return actions

def random_room(seed):
    """Sample distinct tiles. Accept only rooms the reference pair solves for both clues."""
    rng = np.random.default_rng(seed)
    tiles = [(x, y) for y in range(1, 6) for x in range(1, 8)]
    while True:
        cells = [list(tiles[i]) for i in rng.permutation(len(tiles))[:11]]
        config = dict(width=9, height=7, starts=cells[:2], plate=cells[2],
                      switches=cells[3:5], exit=cells[5], walls=cells[6:], max_steps=100)
        room = EscapeRoom(config=config)
        for clue in (0, 1):
            obs, _ = room.reset(seed, clue=clue)
            while not room.done:
                obs, *_ = room.step(scripted(obs, config['walls']))
            if room.outcome != 'success':
                break
        else:
            return config
