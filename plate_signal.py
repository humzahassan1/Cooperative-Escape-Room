"""Presentation variant: B can read the clue only after reaching the plate."""
import copy
import json
from pathlib import Path
import numpy as np
from environment import EscapeRoom, random_room
from options_rl import navigate


class PlateRoom(EscapeRoom):
    clue_holder = 1

    def __init__(self, communication=True, config=None):
        # The fixed room puts A on the plate side, so B starts after a swap.
        # Random layouts set this per seed, before reset runs.
        self.reverse_starts = config is None
        self.layout_seed = None
        super().__init__(communication, config)

    def reset(self, seed=None, clue=None):
        self.has_clue = False
        self.config['plate_holder'] = 1
        obs, info = super().reset(seed, clue)
        if self.reverse_starts:
            self.pos.reverse()
            obs = self.observations()
        return obs, info

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


def _episode_succeeds(config, seed, clue, reverse, tables):
    room = PlateRoom(config=copy.deepcopy(config))
    room.reverse_starts = reverse
    obs, _ = room.reset(seed, clue)
    walls = room.config.get('walls', ())
    while not room.done:
        obs, *_ = room.step(plate_actions(tables, obs, walls))
    return room.outcome == 'success'


def open_plate_room(seed, clue=None, tables=None, attempts=64):
    """Open a randomized vault the plate demo can escape.

    The drawn clue and the original start order are tried first. A seed that
    traps one clue uses the other clue. A seed that traps both is skipped.
    """
    tables = load_tables() if tables is None else tables
    requested = None if clue is None else int(clue)
    for offset in range(attempts):
        trial = int(seed) + offset
        config = random_room(trial)
        if requested is None:
            drawn = int(np.random.default_rng(trial).integers(0, 2))
            candidates = (drawn, 1 - drawn)
        else:
            candidates = (requested,)
        for candidate in candidates:
            for reverse in (False, True):
                if not _episode_succeeds(config, trial, candidate, reverse, tables):
                    continue
                room = PlateRoom(config=copy.deepcopy(config))
                room.reverse_starts = reverse
                room.layout_seed = trial
                obs, _ = room.reset(trial, candidate)
                return room, obs, trial
    raise RuntimeError(f'no plate-signal layout within {attempts} seeds of {seed}')


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
