"""Presentation variant: B can read the clue only after reaching the plate."""
import json
from pathlib import Path
import numpy as np
from environment import EscapeRoom
from options_rl import navigate


class PlateRoom(EscapeRoom):
    clue_holder = 1

    def reset(self, seed=None, clue=None):
        self.has_clue = False
        self.config['plate_holder'] = 1
        obs, info = super().reset(seed, clue)
        self.pos.reverse()  # B starts on the plate side in this fixed-room demo.
        return self.observations(), info

    def interact(self, pairs):
        if self.pos[1] == tuple(self.config['plate']):
            self.has_clue = True
        return super().interact(pairs)

    def observations(self):
        obs = super().observations()
        if not self.has_clue:
            obs['B'][16:18] = 0
        return obs


def load_tables():
    path = Path(__file__).parent / 'runs/options/seed7_comm.json'
    return [np.asarray(t) for t in json.loads(path.read_text())['tables']]


def plate_actions(tables, obs, walls):
    a, b = obs['A'], obs['B']
    has_clue = bool(b[16:18].any())
    received = int(a[-3:].argmax())
    symbol = int(tables[0][b[16:18].argmax()].argmax()) + 1 if has_clue else 0
    if b[12]:
        a_goal, b_goal = tables[4][0].argmax(), tables[3][0].argmax()
    else:
        a_goal, b_goal = tables[2][received].argmax(), 0
    a_move = navigate(a,0,a_goal,walls) if a[12] or received else 4
    return {'A':(a_move,0), 'B':(navigate(b,1,b_goal,walls),symbol)}
