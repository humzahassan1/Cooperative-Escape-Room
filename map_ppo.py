"""Shared PPO with a complete symbolic overhead view and primitive actions."""
import json
from pathlib import Path
import numpy as np
import torch
from environment import EscapeRoom, IDS, random_room
from learning import Policy, compute_gae


class MapRoom(EscapeRoom):
    def reset(self, seed=None, clue=None):
        holder_stream = np.random.SeedSequence(seed).spawn(1)[0]
        self.clue_holder = int(np.random.default_rng(holder_stream).integers(2))
        self.terrain = np.zeros((5,7,9),np.float32)
        self.terrain[0,[0,-1],:] = 1
        self.terrain[0,:,[0,-1]] = 1
        for x,y in self.config.get('walls',()):
            self.terrain[0,y,x] = 1
        for layer,(x,y) in enumerate([self.config['plate'],*self.config['switches'],self.config['exit']],1):
            self.terrain[layer,y,x] = 1
        return super().reset(seed,clue)

    def observations(self):
        obs = super().observations()
        agents = np.zeros((2,7,9),np.float32)
        for i,(x,y) in enumerate(self.pos):
            if not self.escaped[i]:
                agents[i,y,x] = 1
        for i,k in enumerate(IDS):
            # Same encoding for either identity. Own position plane always comes first.
            extras = obs[k][[12,13+i,14-i,15,16,17,18,19,20]]
            obs[k] = np.concatenate((self.terrain.ravel(),agents[[i,1-i]].ravel(),extras))
        return obs


SIZE = 7*7*9+9
ROOT = Path(__file__).parent/'runs'/'map_ppo'

def room_for(seed):
    return MapRoom(config=dict(random_room(seed),interchangeable=True))

def batch(obs):
    return torch.from_numpy(np.stack([[o[k] for k in IDS] for o in obs]))

def load_map():
    model = Policy(SIZE)
    model.load_state_dict(torch.load(ROOT/'policy.pt',weights_only=True))
    return model.eval()

@torch.no_grad()
def moves(model, observations):
    actions = model.sample(batch([observations]))[0][0].tolist()
    return dict(zip(IDS,actions))

def train():
    torch.manual_seed(7)
    rng=np.random.default_rng(7)
    pool=[room_for(i).config for i in range(128)]
    def fresh():
        env=MapRoom(config=pool[rng.integers(len(pool))])
        obs,_=env.reset(int(rng.integers(2**31)))
        return env,obs
    model=Policy(SIZE)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    envs,obs=map(list,zip(*(fresh() for _ in range(16))))
    for update in range(64):
        observations=[]; actions=[]; logs=[]; values=[]; rewards=[]; boot=[]; ends=[]
        for step in range(64):
            x=batch(obs)
            with torch.no_grad():
                a,lp,_,v=model.sample(x)
            observations.append(x); actions.append(a); logs.append(lp); values.append(v)
            next_obs=[]; terminal=[]; finished=[]; r=[]
            for j,pair in enumerate(a.tolist()):
                o,team,t,tr,_=envs[j].step(dict(zip(IDS,pair)))
                r.append([team['A']]*2); terminal.append(t); finished.append(t or tr)
                next_obs.append(o)
                if t or tr: envs[j],obs[j]=fresh()
                else: obs[j]=o
            with torch.no_grad():
                nv=model.critic(model.body(batch(next_obs))).squeeze(-1)
                boot.append(nv*(~torch.tensor(terminal))[:,None])
            rewards.append(torch.tensor(r)); ends.append(torch.tensor(finished).float())
        adv,ret=compute_gae(torch.stack(rewards),torch.stack(values),torch.stack(boot),torch.stack(ends))
        x=torch.cat(observations).reshape(-1,SIZE); a=torch.cat(actions).reshape(-1,2)
        old=torch.cat(logs).flatten(); adv=adv.flatten(); ret=ret.flatten()
        for epoch in range(4):
            for idx in torch.randperm(len(x)).chunk(4):
                _,lp,h,v=model.sample(x[idx],a[idx])
                ratio=(lp-old[idx]).exp()
                ga=adv[idx]; ga=(ga-ga.mean())/(ga.std()+1e-8)
                loss=-torch.minimum(ratio*ga,ratio.clamp(.8,1.2)*ga).mean()+.5*(v-ret[idx]).square().mean()-.01*h.mean()
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),.5); optimizer.step()
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),ROOT/'policy.pt')

@torch.no_grad()
def evaluate():
    model=load_map()
    torch.manual_seed(123)
    rooms=[]; observations=[]
    for seed in range(18000,18025):
        config=room_for(seed).config
        for clue in (0,1):
            for holder in (0,1):
                env=MapRoom(config=config)
                env.reset(seed,clue); env.clue_holder=holder
                rooms.append(env); observations.append(env.observations())
    for step in range(100):
        actions=model.sample(batch(observations))[0].tolist()
        for j,env in enumerate(rooms):
            if not env.done:
                observations[j],*_=env.step(dict(zip(IDS,actions[j])))
    stats=dict(train_seed=7,train_layouts=128,train_steps=65536,test_layouts=25,
               episodes=len(rooms),unlocked=sum(e.unlocked for e in rooms),
               escaped=sum(e.outcome=='success' for e in rooms))
    (ROOT/'evaluation.json').write_text(json.dumps(stats,indent=2))
    print(stats)

if __name__=='__main__':
    train()
    evaluate()
