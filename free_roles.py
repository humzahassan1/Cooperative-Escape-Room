"""Shared reactive controller trained by a (1 + lambda) evolutionary strategy."""
import json
from pathlib import Path
import numpy as np
from environment import EscapeRoom, IDS, MOVES


def setup(seed, clue=None, holder=None, swap=None):
    rng = np.random.default_rng(seed)
    config = EscapeRoom().config.copy()
    config['interchangeable'] = True
    if swap if swap is not None else rng.integers(2):
        config['starts'] = config['starts'][::-1]
    room = EscapeRoom(config=config)
    room.clue_holder = int(rng.integers(2)) if holder is None else holder
    obs, _ = room.reset(seed, clue)
    return room, obs


# A row depends only on this agent's tile, the visible lock and its own
# information. Private clues and received symbols have separate table entries.
CELLS = 63
ROWS = CELLS * 2 * 5


def index(cell, opened, information):
    return cell + CELLS * (opened + 2 * np.where(opened,0,information))


class Policy:
    def __init__(self, genome):
        self.genome = np.asarray(genome)

    def act(self, observations):
        actions = {}
        for i, name in enumerate(IDS):
            obs = observations[name]
            x, y = np.rint(obs[2*i:2*i+2] * [8, 6]).astype(int)
            knows = obs[16:18].any()
            clue = obs[16:18].argmax()
            information = clue + 3 if knows else obs[-3:].argmax()
            action = self.genome[index(x+9*y, int(obs[12]), information)]
            symbol = self.genome[ROWS+clue] if knows else 0
            actions[name] = int(action), int(symbol)
        return actions


def fitness(population, practice=False):
    """Batch the fixed-room transitions for training. Verify saved policies in EscapeRoom."""
    count = len(population)
    rows = np.arange(count)[:,None]
    clue = np.array([0,1,0,1])
    holder = np.array([0,0,1,1])
    positions = [np.full((count,4),29), np.full((count,4),32)]
    # (2,3), (5,3), plate (2,2), switches (5,2)/(5,4), exit (7,3).
    plate, exit_cell = 20,34
    switches = np.array([23,41])
    opened = np.zeros((count,4),bool)
    done = np.zeros_like(opened)
    left = [np.zeros_like(done),np.zeros_like(done)]
    score = np.zeros_like(opened,dtype=float)
    nearest = np.full_like(score,24.)
    shifts = np.array([-9,9,-1,1,0,0])
    horizon = 32 if practice else 100
    for t in range(horizon):
        active = ~done
        acts, proposed = [], []
        for i,pos in enumerate(positions):
            received = population[rows,ROWS+clue] if t else 0
            info = np.where(holder==i,clue+3,received)
            action = population[rows,index(pos,opened,info)]
            dst = pos + shifts[action]
            valid = (dst%9>0)&(dst%9<8)&(dst//9>0)&(dst//9<6)&((dst!=exit_cell)|opened)
            proposed.append(np.where(valid & ~left[i] & active,dst,pos))
            acts.append(action)
        overlap = (proposed[0]==proposed[1]) | ((proposed[0]==positions[1])&(proposed[1]==positions[0]))
        collision = overlap & ~left[0] & ~left[1]
        new = [np.where(collision,pos,dst) for pos,dst in zip(positions,proposed)]
        changed = (new[0]!=positions[0]) | (new[1]!=positions[1])
        for i in range(2):
            j=1-i
            attempt = active & ~opened & (new[i]==plate) & (acts[j]==5)
            correct = attempt & (new[j]==switches[clue])
            wrong = attempt & (new[j]==switches[1-clue])
            opened |= correct
            changed |= correct
            done |= wrong
            score += correct.astype(float) - wrong.astype(float)
        for i in range(2):
            left[i] |= opened & (new[i]==exit_cell)
        if practice:
            distance = sum(abs(p%9-7)+abs(p//9-3) for p in new)
            nearest = np.where(opened,np.minimum(nearest,distance),nearest)
        success = active & left[0] & left[1]
        score += 5*success - .01*active
        done |= success
        # After the first symbol arrives, an unchanged state repeats forever.
        stuck = ~done & ~changed & (t>=1)
        score -= .01*(99-t)*stuck
        done |= stuck
        positions = new
        if done.all():
            break
    if practice:
        score -= (~done)*.01*(100-horizon)
        score += np.where(opened,(24-nearest)*.12,0)
    return score


def train(seed=7):
    rng = np.random.default_rng(seed)
    parent = rng.integers(6,size=ROWS+2,dtype=np.int16)
    parent[-2:] = rng.integers(3,size=2)
    trace = []
    for generation in range(1500):
        population = np.tile(parent,(256,1))
        rate = [.02,.08,.25][generation%3]
        mutation = rng.random(population.shape)<rate
        candidates = rng.integers(6,size=population.shape,dtype=np.int16)
        candidates[:,-2:] %= 3
        population[mutation] = candidates[mutation]
        population[0] = parent
        scores = fitness(population,practice=True)
        averages = scores.mean(axis=1)
        best = rng.choice(np.flatnonzero(np.isclose(averages,averages.max(),atol=1e-10,rtol=0)))
        parent = population[best].copy()
        if generation%50==0:
            trace.append([generation,float(averages[best])])
        if np.min(scores[best])>6:
            break
    return parent, trace, generation+1


def trials(genome, silence=False):
    records = []
    policy = Policy(genome)
    for swapped in (False, True):
        for holder in (0,1):
            for clue in (0,1):
                env,obs = setup(0,clue,holder,swapped)
                env.communication = not silence
                plate_agent = None
                total = 0
                while not env.done:
                    locked = not env.unlocked
                    obs,reward,*_ = env.step(policy.act(obs))
                    total += reward['A']
                    if locked and env.unlocked:
                        plate_agent = IDS[env.pos.index(tuple(env.config['plate']))]
                records.append(dict(swapped=swapped,holder=IDS[holder],clue=clue+1,
                                    plate_agent=plate_agent,outcome=env.outcome,steps=env.steps,reward=total))
    return records


if __name__=='__main__':
    out = Path('runs/free_roles')
    out.mkdir(parents=True,exist_ok=True)
    for seed in (7,8,9):
        genome,trace,generations = train(seed)
        enabled = trials(genome)
        expected = fitness(genome[None])[0]
        np.testing.assert_allclose(expected,[r['reward'] for r in enabled[:4]],atol=1e-10)
        results = dict(seed=seed,generations=generations,genome=genome.tolist(),
                       trace=trace,enabled=enabled,silenced=trials(genome,True))
        (out/f'seed{seed}.json').write_text(json.dumps(results,indent=2))
        print(seed,generations,[r['outcome'] for r in enabled],flush=True)
