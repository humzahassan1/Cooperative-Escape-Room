"""Shared PPO for role coordination on the full public map.

Neither agent has an assigned job. Both see the same board, either may hold
the plate or activate a switch, and each has its own GRU memory. Messages are
silence plus four anonymous symbols. Nothing in the training code gives those
symbols a meaning.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from environment import EscapeRoom, IDS, random_room, scripted
from learning import compute_gae

torch.set_num_threads(1)

MESSAGE_COUNT = 5
SILENCE = 0
CHANNELS = 7
HEIGHT = 7
WIDTH = 9
BOARD = CHANNELS * HEIGHT * WIDTH
EXTRAS = 11
OBS_SIZE = BOARD + EXTRAS
HIDDEN = 64
ROOT = Path(__file__).parent / 'runs' / 'role_coordination'

# Fixed before the final held-out test. These cutoffs decide how the message
# diagnostic is described. They are not tuned on the test layouts.
MIN_INTERPRETATION_COUNT = 5
INTENT_ACCURACY_MARGIN = 0.05
INTENT_MIN_LABELED = 30
INTENT_MIN_BRANCHES = 20
INTENT_COMPLEMENT_RATE = 0.20
CLUE_ACCURACY = 0.70

TRAIN_LAYOUTS = range(0, 1024)
VAL_LAYOUTS = range(20000, 20025)
TEST_LAYOUTS = range(21000, 21100)
PAIRED_SEARCH = range(22000, 27000)
PAIRED_TARGET = 50


def stage_for(progress):
    """Curriculum fraction of the training budget. Evaluation never calls this."""
    if progress < 0.20:
        return 'exit'
    if progress < 0.50:
        return 'near'
    return 'full'


def shaped_reward(stage, paid, escaped_before, escaped_after, reward):
    """One-time team bonus when the first agent exits. Absent from the final stage."""
    if stage in ('exit', 'near') and not paid and escaped_after and not escaped_before:
        return reward + 1.0, True
    return reward, paid


def mix_seed(*parts):
    value = 0x9E3779B1
    for part in parts:
        value = (value ^ (int(part) + 0x9E3779B9)) * 0x85EBCA6B & 0x7FFFFFFF
    return int(value)


def case_generator(layout_seed, clue, holder, swap, stream=17):
    generator = torch.Generator()
    generator.manual_seed(mix_seed(layout_seed, clue, holder, int(bool(swap)), stream))
    return generator


def stream_rngs(seed):
    spawned = np.random.SeedSequence(int(seed)).spawn(3)
    return [np.random.default_rng(item) for item in spawned]


def interior_tiles(config):
    walls = {tuple(p) for p in config.get('walls', ())}
    width, height = config['width'], config['height']
    return [(x, y) for y in range(1, height - 1) for x in range(1, width - 1) if (x, y) not in walls]


def reachable_from(config, origin):
    walls = {tuple(p) for p in config.get('walls', ())}
    width, height = config['width'], config['height']
    start = tuple(origin)
    seen = {start}
    queue = [start]
    while queue:
        x, y = queue.pop()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = (x + dx, y + dy)
            if nxt in seen or nxt in walls:
                continue
            if not (0 < nxt[0] < width - 1 and 0 < nxt[1] < height - 1):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_tile(tiles, target, rng):
    best = min(manhattan(tile, target) for tile in tiles)
    choices = [tile for tile in tiles if manhattan(tile, target) == best]
    return choices[int(rng.integers(len(choices)))]


def mechanisms(config):
    return {tuple(config['plate']), tuple(config['switches'][0]), tuple(config['switches'][1]), tuple(config['exit'])}


def sample_exit_starts(config, rng):
    exit_pos = tuple(config['exit'])
    choices = [tile for tile in reachable_from(config, exit_pos) if tile != exit_pos]
    if len(choices) < 2:
        choices = [tile for tile in interior_tiles(config) if tile != exit_pos]
    picked = rng.choice(len(choices), size=2, replace=False)
    return [choices[int(i)] for i in picked]


def sample_near_starts(config, rng):
    """One legal tile near the plate and one near a switch. The caller assigns them at random."""
    blocked = mechanisms(config)
    legal = [tile for tile in interior_tiles(config) if tile not in blocked]
    if len(legal) < 2:
        raise RuntimeError('room does not have two legal practice starts')
    switch = tuple(config['switches'][int(rng.integers(2))])
    near_plate = nearest_tile(legal, tuple(config['plate']), rng)
    rest = [tile for tile in legal if tile != near_plate]
    near_switch = nearest_tile(rest, switch, rng)
    pair = [near_plate, near_switch]
    if int(rng.integers(2)):
        pair.reverse()
    return pair


def solves(config, clue, seed=0):
    room = EscapeRoom(config=config)
    obs, _ = room.reset(seed, clue=clue)
    walls = config.get('walls', ())
    while not room.done:
        obs, *_ = room.step(scripted(obs, walls))
    return room.outcome == 'success'


def exchange_plate_and_switch(config, clue):
    """Move the plate and the correct switch onto each other's tiles. Walls and exit stay put."""
    cfg = json.loads(json.dumps(config))
    plate = list(cfg['plate'])
    switch = list(cfg['switches'][int(clue)])
    cfg['plate'] = switch
    cfg['switches'][int(clue)] = plate
    return cfg


class RoleRoom(EscapeRoom):
    """Full-map room. Either agent can hold the plate. Messages are silence or four symbols."""

    def __init__(self, communication=True, config=None):
        self.plate_holder = None
        self.switch_activator = None
        self.stage = 'full'
        self.terrain = None
        self.clue_holder = 0
        super().__init__(communication=communication, config=config, message_count=MESSAGE_COUNT)

    def _build_terrain(self):
        terrain = np.zeros((5, HEIGHT, WIDTH), np.float32)
        terrain[0, 0, :] = 1
        terrain[0, -1, :] = 1
        terrain[0, :, 0] = 1
        terrain[0, :, -1] = 1
        for x, y in self.config.get('walls', ()):
            terrain[0, y, x] = 1
        for layer, (x, y) in enumerate([self.config['plate'], *self.config['switches'], self.config['exit']], 1):
            terrain[layer, y, x] = 1
        self.terrain = terrain

    def reset(self, seed=None, clue=None, holder=None, swap_starts=None, stage='full'):
        self.stage = stage
        self.plate_holder = None
        self.switch_activator = None
        self._build_terrain()
        super().reset(seed, clue)
        holder_rng, swap_rng, place_rng = stream_rngs(0 if seed is None else int(seed))
        self.clue_holder = int(holder_rng.integers(2)) if holder is None else int(holder)
        if stage == 'full':
            swap = bool(swap_rng.integers(2)) if swap_starts is None else bool(swap_starts)
            if swap:
                self.pos = [self.pos[1], self.pos[0]]
            self.unlocked = False
        elif stage == 'exit':
            self.unlocked = True
            self.pos = list(sample_exit_starts(self.config, place_rng))
        elif stage == 'near':
            self.unlocked = False
            self.pos = list(sample_near_starts(self.config, place_rng))
        else:
            raise ValueError(f'unknown stage {stage}')
        return self.observations(), self.info()

    def observations(self):
        result = {}
        for i, name in enumerate(IDS):
            agents = np.zeros((2, HEIGHT, WIDTH), np.float32)
            for j, (x, y) in enumerate(self.pos):
                if not self.escaped[j]:
                    agents[0 if j == i else 1, y, x] = 1
            board = np.concatenate((self.terrain, agents), 0)
            clue = np.zeros(2, np.float32)
            if i == self.clue_holder:
                clue[int(self.clue)] = 1
            message = np.zeros(MESSAGE_COUNT, np.float32)
            message[int(self.messages[1 - i])] = 1
            extras = np.array([
                float(self.unlocked), float(self.escaped[i]), float(self.escaped[1 - i]),
                1 - self.steps / self.config['max_steps'], *clue, *message,
            ], np.float32)
            result[name] = np.concatenate((board.ravel(), extras))
        return result

    def step(self, actions):
        was_unlocked = self.unlocked
        out = super().step(actions)
        if self.unlocked and not was_unlocked:
            plate = tuple(self.config['plate'])
            correct = tuple(self.config['switches'][self.clue])
            for holder in (0, 1):
                partner = 1 - holder
                move = int(actions[IDS[partner]][0])
                if tuple(self.pos[holder]) == plate and tuple(self.pos[partner]) == correct and move == 5:
                    self.plate_holder = holder
                    self.switch_activator = partner
                    break
        return out

    def clone(self):
        other = RoleRoom(self.communication, config=self.config)
        other.clue = int(self.clue)
        other.clue_holder = int(self.clue_holder)
        other.pos = [tuple(p) for p in self.pos]
        other.escaped = list(self.escaped)
        other.messages = list(self.messages)
        other.unlocked = bool(self.unlocked)
        other.steps = int(self.steps)
        other.done = bool(self.done)
        other.outcome = self.outcome
        other.stage = self.stage
        other.plate_holder = self.plate_holder
        other.switch_activator = self.switch_activator
        other.terrain = self.terrain
        return other


def split_obs(flat):
    board = flat[..., :BOARD].reshape(-1, CHANNELS, HEIGHT, WIDTH)
    extras = flat[..., BOARD:]
    return board, extras


def apply_done(hidden, done):
    """Zero selected memories. A 1-D done mask over environments clears both agents."""
    mask = 1.0 - done.to(dtype=hidden.dtype)
    if hidden.ndim == done.ndim + 2:
        mask = mask[:, None, None]
    elif hidden.ndim == done.ndim + 1:
        mask = mask.unsqueeze(-1)
    else:
        raise ValueError('done mask does not align with hidden state')
    return hidden * mask


def categorical_sample(logits, generator=None):
    """Gumbel-max sample. One uniform per logit, so forked generators stay aligned."""
    uniforms = torch.rand(logits.shape, generator=generator)
    gumbel = -torch.log(-torch.log(uniforms.clamp(1e-8, 1.0 - 1e-8)))
    return torch.argmax(logits + gumbel, dim=-1)


class RolePolicy(nn.Module):
    """One set of weights. Callers keep a separate hidden state per agent."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.project = nn.Linear(32 * HEIGHT * WIDTH, HIDDEN)
        self.fuse = nn.Linear(HIDDEN + EXTRAS, HIDDEN)
        self.gru = nn.GRUCell(HIDDEN, HIDDEN)
        self.actor = nn.Linear(HIDDEN, 6)
        self.message = nn.Linear(HIDDEN, MESSAGE_COUNT)
        self.critic = nn.Linear(HIDDEN, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
        for name, param in self.gru.named_parameters():
            if param.dim() >= 2:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.message.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def encode(self, board, extras):
        features = self.conv(board).flatten(1)
        spatial = torch.tanh(self.project(features))
        return torch.tanh(self.fuse(torch.cat((spatial, extras), -1)))

    def heads(self, hidden):
        return self.actor(hidden), self.message(hidden), self.critic(hidden).squeeze(-1)


def sequence_forward(model, boards, extras, actions, resets, h0):
    """Replay contiguous rollout sequences so the GRU receives gradients through time."""
    length, batch = actions.shape[:2]
    encoded = model.encode(boards.reshape(length * batch, *boards.shape[2:]), extras.reshape(length * batch, EXTRAS))
    encoded = encoded.reshape(length, batch, HIDDEN)
    hidden = h0
    logps, entropies, values = [], [], []
    for t in range(length):
        if t:
            hidden = apply_done(hidden, resets[t])
        hidden = model.gru(encoded[t], hidden)
        move_logits, message_logits, value = model.heads(hidden)
        moves = Categorical(logits=move_logits)
        messages = Categorical(logits=message_logits)
        logps.append(moves.log_prob(actions[t, :, 0]) + messages.log_prob(actions[t, :, 1]))
        entropies.append(moves.entropy() + messages.entropy())
        values.append(value)
    return torch.stack(logps), torch.stack(entropies), torch.stack(values)


def stack_observations(observations):
    return torch.from_numpy(np.stack([np.stack([obs[name] for name in IDS]) for obs in observations]))


def policy_forward(model, observations, hidden):
    """observations: list of dicts or a tensor (N, 2, OBS). hidden: (N, 2, H)."""
    if isinstance(observations, list):
        flat = stack_observations(observations)
    else:
        flat = observations
    count = flat.shape[0]
    board, extras = split_obs(flat.reshape(-1, OBS_SIZE))
    encoded = model.encode(board, extras)
    updated = model.gru(encoded, hidden.reshape(-1, HIDDEN)).reshape(count, 2, HIDDEN)
    move_logits, message_logits, value = model.heads(updated.reshape(-1, HIDDEN))
    move_logits = move_logits.reshape(count, 2, 6)
    message_logits = message_logits.reshape(count, 2, MESSAGE_COUNT)
    value = value.reshape(count, 2)
    return move_logits, message_logits, value, updated


def sample_actions(move_logits, message_logits, generator):
    actions = {}
    for index, name in enumerate(IDS):
        move = int(categorical_sample(move_logits[index], generator))
        message = int(categorical_sample(message_logits[index], generator))
        actions[name] = (move, message)
    return actions


def plain_config(config):
    return json.loads(json.dumps(config))


def load_layouts(seeds=TRAIN_LAYOUTS, cache=ROOT / 'train_layouts.json'):
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    layouts = [plain_config(random_room(int(seed))) for seed in seeds]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(layouts))
    return layouts


def make_room(config, communication):
    return RoleRoom(communication=communication, config=dict(config, interchangeable=True))


def reset_for_evaluation(config, clue, holder, swap, communication=True, seed=0):
    """Complete locked puzzle, original rewards. Curriculum bonuses are not applied here."""
    env = make_room(config, communication)
    env.reset(seed, clue=clue, holder=holder, swap_starts=None if swap is None else bool(swap), stage='full')
    return env


def evaluation_cases(seeds):
    cases = []
    for seed in seeds:
        config = plain_config(random_room(int(seed)))
        for clue in (0, 1):
            for holder in (0, 1):
                for swap in (0, 1):
                    cases.append(dict(config=config, layout_seed=int(seed), clue=clue, holder=holder, swap=swap, stream=17))
    return cases


def episode_record(env, case, actions, keep_actions=False):
    record = dict(
        layout_seed=int(case['layout_seed']), clue=int(case['clue']), holder=int(case['holder']),
        swap=bool(case['swap']), outcome=env.outcome, steps=int(env.steps), unlocked=bool(env.unlocked),
        plate_holder=None if env.plate_holder is None else int(env.plate_holder),
        switch_activator=None if env.switch_activator is None else int(env.switch_activator),
    )
    if keep_actions:
        record['actions'] = actions
        record['config'] = plain_config(dict(env.config, interchangeable=True))
    return record


@torch.no_grad()
def run_cases(model, cases, communication=True):
    """Batched evaluation. Each episode keeps the generator used by a solo replay."""
    model.eval()
    envs, generators, hidden, logs = [], [], [], []
    for case in cases:
        env = reset_for_evaluation(case['config'], case['clue'], case['holder'], case['swap'], communication, case['layout_seed'])
        envs.append(env)
        generators.append(case_generator(case['layout_seed'], case['clue'], case['holder'], case['swap'], case.get('stream', 17)))
        hidden.append(torch.zeros(2, HIDDEN))
        logs.append([])
    pending = set(range(len(envs)))
    while pending:
        index = sorted(pending)
        observations = [envs[i].observations() for i in index]
        state = torch.stack([hidden[i] for i in index])
        move_logits, message_logits, _, updated = policy_forward(model, observations, state)
        for row, i in enumerate(index):
            actions = sample_actions(move_logits[row], message_logits[row], generators[i])
            logs[i].append({name: [int(value) for value in pair] for name, pair in actions.items()})
            envs[i].step(actions)
            hidden[i] = updated[row].detach()
            if envs[i].done:
                pending.remove(i)
    return [episode_record(env, case, log) for env, case, log in zip(envs, cases, logs)]


@torch.no_grad()
def play(model, env, generator):
    model.eval()
    hidden = torch.zeros(2, HIDDEN)
    log = []
    while not env.done:
        move_logits, message_logits, _, updated = policy_forward(model, [env.observations()], hidden[None])
        actions = sample_actions(move_logits[0], message_logits[0], generator)
        log.append({name: [int(value) for value in pair] for name, pair in actions.items()})
        env.step(actions)
        hidden = updated[0].detach()
    return log


def summarize_episodes(records, seed, communication, steps):
    successes = [item for item in records if item['outcome'] == 'success']
    outcomes = {}
    for item in records:
        outcomes[item['outcome']] = outcomes.get(item['outcome'], 0) + 1
    roles = []
    for item in records:
        if item['plate_holder'] is None:
            continue
        roles.append(dict(
            layout_seed=item['layout_seed'], clue=item['clue'], holder=item['holder'], swap=item['swap'],
            plate_holder=IDS[item['plate_holder']], switch_activator=IDS[item['switch_activator']],
            outcome=item['outcome'], steps=item['steps'],
        ))
    return dict(
        train_seed=int(seed), communication=bool(communication), train_steps=int(steps),
        episodes=len(records), unlocked=sum(item['unlocked'] for item in records),
        escaped=len(successes),
        unlock_rate=sum(item['unlocked'] for item in records) / len(records),
        escape_rate=len(successes) / len(records),
        mean_success_steps=None if not successes else float(np.mean([item['steps'] for item in successes])),
        outcomes=outcomes, roles=roles,
    )


def save_model(model, path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), **payload), path)


def load_role(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = RolePolicy()
    model.load_state_dict(payload['model'])
    model.eval()
    return model, payload


def train(out, seed=7, steps=1_048_576, communication=True, num_envs=16, rollout=128,
          layouts=None, periodic_eval=True, final_eval=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    layouts = layouts if layouts is not None else load_layouts()
    model = RolePolicy()
    initial = model.actor.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    config = dict(
        algorithm='ppo', communication=bool(communication), total_steps=int(steps), seed=int(seed),
        num_envs=int(num_envs), rollout=int(rollout), epochs=4, minibatches=4, learning_rate=3e-4,
        adam_eps=1e-5, gamma=0.99, gae_lambda=0.95, clip=0.2, entropy=0.01, value_coef=0.5,
        gradient_clip=0.5, hidden_size=HIDDEN, message_count=MESSAGE_COUNT, observation_size=OBS_SIZE,
        curriculum=dict(exit=0.20, near=0.30, full=0.50), first_exit_bonus=1.0,
        bonus_stages=['exit', 'near'], train_layout_seeds=[0, 1023],
        validation_layout_seeds=[20000, 20024], test_layout_seeds=[21000, 21099],
        architecture='shared conv encoder, 64-unit GRU, separate memory, primitive moves, 5 messages',
    )
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'command.txt').write_text(
        f'python role_ppo.py train --seed {seed} --steps {steps}'
        f'{" --no-communication" if not communication else ""} --out {out}\n'
    )

    def fresh(progress):
        layout = layouts[int(rng.integers(len(layouts)))]
        env = make_room(layout, communication)
        episode_seed = int(rng.integers(1, 2**31 - 1))
        obs, _ = env.reset(episode_seed, stage=stage_for(progress))
        return env, obs

    envs, observations = [], []
    for _ in range(num_envs):
        env, obs = fresh(0.0)
        envs.append(env)
        observations.append(obs)
    hidden = torch.zeros(num_envs, 2, HIDDEN)
    bonus_paid = [False] * num_envs
    episode_return = np.zeros(num_envs)
    logs = []
    started = time.perf_counter()
    iterations = int(np.ceil(steps / (num_envs * rollout)))
    global_step = 0
    for iteration in range(1, iterations + 1):
        boards = torch.zeros((rollout, num_envs, 2, CHANNELS, HEIGHT, WIDTH))
        extras = torch.zeros((rollout, num_envs, 2, EXTRAS))
        actions = torch.zeros((rollout, num_envs, 2, 2), dtype=torch.long)
        logp = torch.zeros((rollout, num_envs, 2))
        values = torch.zeros((rollout, num_envs, 2))
        rewards = torch.zeros((rollout, num_envs, 2))
        next_values = torch.zeros((rollout, num_envs, 2))
        ends = torch.zeros((rollout, num_envs))
        h0 = hidden.detach().clone()
        returns = []
        successes = 0
        model.train()
        for t in range(rollout):
            flat = stack_observations(observations)
            boards[t] = flat[..., :BOARD].reshape(num_envs, 2, CHANNELS, HEIGHT, WIDTH)
            extras[t] = flat[..., BOARD:]
            with torch.no_grad():
                move_logits, message_logits, value, updated = policy_forward(model, flat, hidden)
                move = categorical_sample(move_logits)
                message = categorical_sample(message_logits)
                chosen = torch.stack((move, message), -1)
                move_dist = Categorical(logits=move_logits)
                message_dist = Categorical(logits=message_logits)
                logp[t] = move_dist.log_prob(move) + message_dist.log_prob(message)
                values[t] = value
            actions[t] = chosen
            hidden = updated.detach()
            post, terminals, dones = [None] * num_envs, [False] * num_envs, [False] * num_envs
            for j, env in enumerate(envs):
                escaped_before = any(env.escaped)
                stage = env.stage
                pair = {name: (int(chosen[j, i, 0]), int(chosen[j, i, 1])) for i, name in enumerate(IDS)}
                obs, reward, terminal, truncated, _ = env.step(pair)
                reward, bonus_paid[j] = shaped_reward(stage, bonus_paid[j], escaped_before, any(env.escaped), reward['A'])
                rewards[t, j] = reward
                episode_return[j] += reward
                post[j] = obs
                terminals[j] = bool(terminal)
                dones[j] = bool(terminal or truncated)
                if dones[j]:
                    returns.append(float(episode_return[j]))
                    episode_return[j] = 0
                    successes += int(env.outcome == 'success')
            ends[t] = torch.tensor(dones, dtype=torch.float32)
            alive = [j for j in range(num_envs) if not terminals[j]]
            if alive:
                with torch.no_grad():
                    _, _, bootstrap, _ = policy_forward(model, [post[j] for j in alive], hidden[alive])
                for row, j in enumerate(alive):
                    next_values[t, j] = bootstrap[row]
            next_observations = []
            for j in range(num_envs):
                if dones[j]:
                    progress = min(global_step / steps, 0.999999)
                    envs[j], obs = fresh(progress)
                    hidden[j] = 0
                    bonus_paid[j] = False
                    next_observations.append(obs)
                else:
                    next_observations.append(post[j])
            observations = next_observations
            global_step += num_envs
        advantage, returns_t = compute_gae(rewards, values, next_values, ends)
        sequence = num_envs * 2
        boards_f = boards.reshape(rollout, sequence, CHANNELS, HEIGHT, WIDTH)
        extras_f = extras.reshape(rollout, sequence, EXTRAS)
        actions_f = actions.reshape(rollout, sequence, 2)
        old_logp = logp.reshape(rollout, sequence)
        adv = advantage.reshape(rollout, sequence)
        ret = returns_t.reshape(rollout, sequence)
        h0_f = h0.reshape(sequence, HIDDEN)
        resets = torch.zeros((rollout, sequence))
        if rollout > 1:
            resets[1:] = ends[:-1].repeat_interleave(2, dim=1)
        losses = []
        for _epoch in range(4):
            for index in torch.randperm(sequence).chunk(4):
                new_logp, entropy, value = sequence_forward(
                    model, boards_f[:, index], extras_f[:, index], actions_f[:, index], resets[:, index], h0_f[index]
                )
                ratio = (new_logp - old_logp[:, index]).exp()
                ga = adv[:, index].reshape(-1)
                ga = (ga - ga.mean()) / (ga.std(unbiased=False) + 1e-8)
                ratio_f = ratio.reshape(-1)
                policy_loss = torch.maximum(-ga * ratio_f, -ga * ratio_f.clamp(0.8, 1.2)).mean()
                value_loss = 0.5 * (value.reshape(-1) - ret[:, index].reshape(-1)).square().mean()
                loss = policy_loss + value_loss - 0.01 * entropy.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite PPO loss')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                losses.append(float(loss.detach()))
        row = dict(
            iteration=iteration, steps=global_step, stage=stage_for(min(global_step / steps, 0.999999)),
            loss=float(np.mean(losses)), mean_curriculum_return=None if not returns else float(np.mean(returns)),
            curriculum_successes=successes, elapsed_seconds=time.perf_counter() - started,
        )
        if periodic_eval and iteration % 64 == 0:
            val_cases = evaluation_cases(range(20000, 20004))
            val_records = run_cases(model, val_cases, communication)
            row['val_episodes'] = len(val_records)
            row['val_unlocked'] = sum(item['unlocked'] for item in val_records)
            row['val_escaped'] = sum(item['outcome'] == 'success' for item in val_records)
        logs.append(row)
        (out / 'training.json').write_text(json.dumps(logs))
        print(json.dumps({key: row[key] for key in row if key != 'roles'}), flush=True)
    if torch.equal(initial, model.actor.weight):
        raise RuntimeError('policy weights did not change during training')
    payload = dict(config=config, steps=global_step, seed=int(seed), communication=bool(communication),
                   observation_size=OBS_SIZE, message_count=MESSAGE_COUNT, hidden_size=HIDDEN)
    save_model(model, out / 'checkpoint.pt', payload)
    result = None
    if final_eval:
        result = evaluate(model, seed, communication, global_step, out)
    return result


def evaluate(model, seed, communication, steps, out=None):
    records = run_cases(model, evaluation_cases(TEST_LAYOUTS), communication)
    summary = summarize_episodes(records, seed, communication, steps)
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / 'evaluation.json').write_text(json.dumps(summary, indent=2))
        best = select_replay(records, model, communication)
        if best is not None:
            (Path(out) / 'replay_candidate.json').write_text(json.dumps(best))
    print(json.dumps({key: summary[key] for key in summary if key != 'roles'}), flush=True)
    return summary


def select_replay(records, model, communication):
    """Keep one reproducible episode. Prefer a short success, then an unlock, then a timeout."""
    def rank(item):
        return (item['outcome'] == 'success', item['unlocked'], -item['steps'] if item['outcome'] == 'success' else 0)
    if not records:
        return None
    chosen = max(records, key=rank)
    env = reset_for_evaluation(random_room(chosen['layout_seed']), chosen['clue'], chosen['holder'], chosen['swap'], communication, chosen['layout_seed'])
    actions = play(model, env, case_generator(chosen['layout_seed'], chosen['clue'], chosen['holder'], chosen['swap']))
    if env.outcome != chosen['outcome'] or env.steps != chosen['steps']:
        raise RuntimeError('solo replay does not match batched evaluation')
    return dict(
        layout_seed=chosen['layout_seed'], clue=chosen['clue'], holder=chosen['holder'], swap=chosen['swap'],
        outcome=env.outcome, steps=env.steps, unlocked=env.unlocked,
        plate_holder=None if env.plate_holder is None else int(env.plate_holder),
        switch_activator=None if env.switch_activator is None else int(env.switch_activator),
        config=plain_config(dict(env.config, interchangeable=True)), actions=actions,
        note='action trace of one held-out episode',
    )


def build_pairs(target=PAIRED_TARGET):
    rng = np.random.default_rng(22000)
    pairs = []
    for seed in PAIRED_SEARCH:
        if len(pairs) >= target:
            break
        base = plain_config(random_room(int(seed)))
        clue = int(rng.integers(2))
        holder = int(rng.integers(2))
        swap = int(rng.integers(2))
        other = exchange_plate_and_switch(base, clue)
        if not (solves(other, 0) and solves(other, 1)):
            continue
        pairs.append(dict(layout_seed=int(seed), clue=clue, holder=holder, swap=swap, base=base, other=other))
    return pairs


def paired_roles(model, communication, out=None):
    pairs = build_pairs()
    cases = []
    for pair in pairs:
        for config in (pair['base'], pair['other']):
            cases.append(dict(config=config, layout_seed=pair['layout_seed'], clue=pair['clue'], holder=pair['holder'], swap=pair['swap'], stream=17))
    records = run_cases(model, cases, communication)
    episodes = []
    both_solved = one_solved = neither = job_changed = job_changed_and_escaped = 0
    showcase = None
    for index, pair in enumerate(pairs):
        left, right = records[2 * index], records[2 * index + 1]
        solved = [left['outcome'] == 'success', right['outcome'] == 'success']
        both_solved += int(all(solved))
        one_solved += int(sum(solved) == 1)
        neither += int(not any(solved))
        changed = left['unlocked'] and right['unlocked'] and left['plate_holder'] != right['plate_holder']
        job_changed += int(bool(changed))
        job_changed_and_escaped += int(bool(changed) and all(solved))
        row = dict(
            layout_seed=pair['layout_seed'], clue=pair['clue'], holder=pair['holder'], swap=bool(pair['swap']),
            left=left, right=right, both_escaped=all(solved), job_changed=bool(changed),
        )
        episodes.append(row)
        if changed and (showcase is None or (all(solved) and not showcase['both_escaped'])):
            showcase = row
    summary = dict(
        pairs=len(pairs), denominator=len(pairs), both_solved=both_solved, one_solved=one_solved,
        neither_solved=neither, job_changed=job_changed, job_changed_and_both_escaped=job_changed_and_escaped,
        episodes=[{key: value for key, value in item.items()} for item in episodes],
    )
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        public = dict(summary)
        public['episodes'] = [{
            'layout_seed': item['layout_seed'], 'clue': item['clue'], 'holder': item['holder'], 'swap': item['swap'],
            'both_escaped': item['both_escaped'], 'job_changed': item['job_changed'],
            'left': {key: item['left'][key] for key in ('outcome', 'steps', 'unlocked', 'plate_holder', 'switch_activator')},
            'right': {key: item['right'][key] for key in ('outcome', 'steps', 'unlocked', 'plate_holder', 'switch_activator')},
        } for item in episodes]
        (out / 'paired.json').write_text(json.dumps(public, indent=2))
        if showcase is not None:
            (out / 'pair_showcase.json').write_text(json.dumps(dict(
                layout_seed=showcase['layout_seed'], clue=showcase['clue'], holder=showcase['holder'], swap=showcase['swap']
            )))
    print(json.dumps({key: summary[key] for key in summary if key != 'episodes'}), flush=True)
    return summary


def on_mechanism(env):
    spots = {tuple(env.config['plate']), tuple(env.config['switches'][0]), tuple(env.config['switches'][1])}
    return any(tuple(pos) in spots for pos in env.pos)


def advance_contact(seen_contact, standing_on_mechanism):
    """Once a plate or switch has been touched, later messages stay ineligible."""
    return bool(seen_contact or standing_on_mechanism)


def signal_features(config, positions, clue, sender):
    plate = tuple(config['plate'])
    correct = tuple(config['switches'][clue])
    wrong = tuple(config['switches'][1 - clue])
    me, other = positions[sender], positions[1 - sender]
    return np.array([
        1.0, float(clue), manhattan(me, plate) / 10, manhattan(me, correct) / 10, manhattan(me, wrong) / 10,
        float(manhattan(me, plate) < manhattan(other, plate)),
        float(manhattan(me, correct) < manhattan(other, correct)),
    ], np.float64)


@torch.no_grad()
def collect_signals(model, cases, communication, snapshots=False, replacement_of=None):
    """Pre-contact messages, sender jobs, and the first eligible intervention point."""
    rows, branches = [], []
    model.eval()
    for case in cases:
        env = reset_for_evaluation(case['config'], case['clue'], case['holder'], case['swap'], communication, case['layout_seed'])
        generator = case_generator(case['layout_seed'], case['clue'], case['holder'], case['swap'], case.get('stream', 17))
        hidden = torch.zeros(2, HIDDEN)
        pending = []
        branch = None
        seen_contact = on_mechanism(env)
        while not env.done:
            before = [tuple(pos) for pos in env.pos]
            eligible = not seen_contact
            move_logits, message_logits, _, updated = policy_forward(model, [env.observations()], hidden[None])
            actions = sample_actions(move_logits[0], message_logits[0], generator)
            env.step(actions)
            hidden = updated[0].detach()
            seen_contact = advance_contact(seen_contact, on_mechanism(env))
            if not eligible or env.done:
                continue
            pending.append(dict(positions=before, messages=list(env.messages), clue=int(env.clue)))
            if snapshots and branch is None and replacement_of is not None:
                for sender in (0, 1):
                    replacement = replacement_of(env.messages[sender], env.clue)
                    if replacement is None:
                        continue
                    branch = dict(
                        case=case, sender=int(sender), symbol=int(env.messages[sender]),
                        replacement=int(replacement[0]), replacement_kind=replacement[1],
                        hidden=hidden.clone(), generator=generator.get_state().clone(), env=env.clone(),
                    )
                    break
        jobs = {}
        if env.plate_holder is not None:
            jobs[env.plate_holder] = 'plate'
        if env.switch_activator is not None:
            jobs[env.switch_activator] = 'switch'
        for item in pending:
            for sender in (0, 1):
                rows.append(dict(
                    clue=item['clue'], symbol=int(item['messages'][sender]), sender=sender,
                    job=jobs.get(sender), features=signal_features(env.config, item['positions'], item['clue'], sender),
                    layout_seed=case['layout_seed'], holder=case['holder'], swap=bool(case['swap']),
                ))
        if branch is not None:
            branch['sender_job'] = jobs.get(branch['sender'])
            branch['receiver_job'] = jobs.get(1 - branch['sender'])
            branch['outcome'] = env.outcome
            branches.append(branch)
    return rows, branches


def fit_logistic(features, labels, steps=400, lr=0.2, l2=1e-2):
    mean = features.mean(0)
    std = features.std(0)
    std[std < 1e-6] = 1
    std[0] = 1
    mean[0] = 0
    x = (features - mean) / std
    y = labels.astype(np.float64)
    weight = np.zeros(x.shape[1])
    for _ in range(steps):
        logits = np.clip(x @ weight, -30, 30)
        probs = 1 / (1 + np.exp(-logits))
        grad = x.T @ (probs - y) / len(y) + l2 * weight
        weight -= lr * grad
    return weight, mean, std


def predict_logistic(weight, mean, std, features):
    x = (features - mean) / std
    return 1 / (1 + np.exp(-np.clip(x @ weight, -30, 30)))


def last_signal_rows(rows):
    """One pre-contact message per sender per episode: the last one before a mechanism."""
    chosen = {}
    for row in rows:
        chosen[(row['layout_seed'], row['clue'], row['holder'], row['swap'], row['sender'])] = row
    return list(chosen.values())


def labeled_matrix(rows, with_symbol):
    usable = [row for row in rows if row['job'] in ('plate', 'switch')]
    if not usable:
        return np.zeros((0, 1)), np.zeros((0,))
    features = [row['features'] for row in usable]
    if with_symbol:
        symbols = np.zeros((len(usable), MESSAGE_COUNT))
        for index, row in enumerate(usable):
            symbols[index, row['symbol']] = 1
        features = [np.concatenate((row['features'], symbols[index])) for index, row in enumerate(usable)]
    labels = np.array([1.0 if row['job'] == 'plate' else 0.0 for row in usable])
    return np.stack(features), labels


def freeze_interpretation(rows):
    counts = {}
    observed = {0: set(), 1: set()}
    support = {}
    for row in rows:
        if row['job'] not in ('plate', 'switch'):
            observed[row['clue']].add(row['symbol'])
            continue
        key = (row['clue'], row['symbol'])
        counts.setdefault(key, {'plate': 0, 'switch': 0})
        counts[key][row['job']] += 1
        observed[row['clue']].add(row['symbol'])
        support[key] = support.get(key, 0) + 1
    # Symbols seen before contact count even when the sender never received a job.
    for row in rows:
        observed[row['clue']].add(row['symbol'])
    frozen = {}
    public = {'0': {}, '1': {}}
    for clue in (0, 1):
        for symbol in range(MESSAGE_COUNT):
            key = (clue, symbol)
            stats = counts.get(key, {'plate': 0, 'switch': 0})
            total = stats['plate'] + stats['switch']
            job = None
            if total >= MIN_INTERPRETATION_COUNT and stats['plate'] != stats['switch']:
                job = 'plate' if stats['plate'] > stats['switch'] else 'switch'
            frozen[key] = job
            public[str(clue)][str(symbol)] = dict(job=job, n=total, plate=stats['plate'], switch=stats['switch'])
    return frozen, observed, public


def replacement_table(observed, frozen):
    def choose(symbol, clue):
        options = [item for item in observed.get(clue, ()) if item != symbol]
        if not options:
            return None
        implied = frozen.get((clue, symbol))
        differing = [item for item in options if frozen.get((clue, item)) not in (None, implied)]
        if differing:
            return differing[0], 'different_job'
        return options[0], 'other_symbol'
    return choose


def opposite(job):
    return {'plate': 'switch', 'switch': 'plate'}.get(job)


@torch.no_grad()
def rollout_from(model, env, hidden, generator, watch, limit=5):
    moves = []
    while not env.done:
        move_logits, message_logits, _, updated = policy_forward(model, [env.observations()], hidden[None])
        actions = sample_actions(move_logits[0], message_logits[0], generator)
        if len(moves) < limit:
            moves.append(actions[IDS[watch]][0])
        env.step(actions)
        hidden = updated[0].detach()
    return env, moves


def restore_generator(state):
    generator = torch.Generator()
    generator.set_state(state.clone())
    return generator


def receiver_job(env, receiver):
    if env.plate_holder == receiver:
        return 'plate'
    if env.switch_activator == receiver:
        return 'switch'
    return None


def intervene(model, branches):
    eligible = len(branches)
    completed = role_changes = disruptions = complementary = move_changes = 0
    different_job = different_role = different_complement = 0
    for branch in branches:
        receiver = 1 - branch['sender']
        original = branch['env'].clone()
        altered = branch['env'].clone()
        altered.messages[branch['sender']] = branch['replacement']
        hidden = branch['hidden'].clone()
        base_env, base_moves = rollout_from(model, original, hidden.clone(), restore_generator(branch['generator']), receiver)
        alt_env, alt_moves = rollout_from(model, altered, hidden.clone(), restore_generator(branch['generator']), receiver)
        completed += 1
        if base_moves != alt_moves:
            move_changes += 1
        success_crash = base_env.outcome == 'success' and alt_env.outcome != 'success'
        base_job = receiver_job(base_env, receiver)
        alt_job = receiver_job(alt_env, receiver)
        changed = (not success_crash and base_env.unlocked and alt_env.unlocked
                   and base_job is not None and alt_job is not None and base_job != alt_job)
        if success_crash:
            disruptions += 1
        elif changed:
            role_changes += 1
        if branch['replacement_kind'] == 'different_job':
            different_job += 1
            if changed:
                different_role += 1
                implied = opposite(branch.get('implied_sender_job'))
                if implied is not None and alt_job == implied:
                    different_complement += 1
                    complementary += 1
    return dict(
        eligible=eligible, completed=completed, receiver_move_changes=move_changes,
        role_changes=role_changes, disruptions=disruptions, complementary=complementary,
        different_job_branches=different_job, different_job_role_changes=different_role,
        different_job_complementary=different_complement,
    )


def accuracy(predicted, labels):
    if len(labels) == 0:
        return None
    return float(np.mean(predicted == labels))


def message_conclusion(stats):
    job_ready = stats['heldout_labeled'] >= INTENT_MIN_LABELED and stats['position_job_accuracy'] is not None
    predicts_job = bool(job_ready and stats['symbol_job_accuracy'] is not None
                        and stats['symbol_job_accuracy'] >= stats['position_job_accuracy'] + INTENT_ACCURACY_MARGIN)
    rate = None
    if stats['different_job_branches']:
        rate = stats['different_job_complementary'] / stats['different_job_branches']
    changes_partner = bool(stats['different_job_branches'] >= INTENT_MIN_BRANCHES and rate is not None
                           and rate >= INTENT_COMPLEMENT_RATE)
    clue = stats['clue_accuracy'] is not None and stats['clue_accuracy'] >= CLUE_ACCURACY
    if predicts_job and changes_partner:
        text = ('The held-out messages meet the pre-registered intent-signal rule: a frozen validation '
                'reading predicts the sender job beyond clue and position, and replacing the symbol '
                'changes the partner complementary job often enough to count.')
        kind = 'intent'
    elif clue and not predicts_job:
        text = ('The evidence supports clue transmission alone. Symbols predict the private clue on '
                'held-out episodes, and they do not meet the rule for an intent signal.')
        kind = 'clue'
    elif not job_ready:
        text = ('Too few held-out episodes assigned a plate holder and a switch activator to test '
                'whether messages predict the sender job. No intent signal is claimed.')
        kind = 'insufficient'
    else:
        text = ('No intent signal is claimed. Messages did not both predict the sender job beyond clue '
                'and position and systematically change the partner complementary job.')
        kind = 'none'
    return dict(kind=kind, text=text, complement_rate=rate, predicts_job=predicts_job, changes_partner=changes_partner)


def diagnose(model, communication, out=None):
    val_cases = evaluation_cases(VAL_LAYOUTS)
    test_cases = evaluation_cases(TEST_LAYOUTS)
    validation_all, _ = collect_signals(model, val_cases, communication, snapshots=False)
    validation_rows = last_signal_rows(validation_all)
    frozen, _, public = freeze_interpretation(validation_rows)
    observed = {0: set(), 1: set()}
    for row in validation_all:
        observed[row['clue']].add(row['symbol'])
    choose = replacement_table(observed, frozen)
    test_all, branches = collect_signals(model, test_cases, communication, snapshots=True, replacement_of=choose)
    test_rows = last_signal_rows(test_all)
    for branch in branches:
        branch['implied_sender_job'] = frozen.get((branch['env'].clue, branch['replacement']))
    intervention = intervene(model, branches)
    vx, vy = labeled_matrix(validation_rows, with_symbol=False)
    tx, ty = labeled_matrix(test_rows, with_symbol=False)
    position_accuracy = None
    symbol_accuracy = None
    if len(vy) >= 2 and len(np.unique(vy)) == 2 and len(ty):
        weight, mean, std = fit_logistic(vx, vy)
        position_accuracy = accuracy((predict_logistic(weight, mean, std, tx) >= 0.5).astype(np.float64), ty)
        mask = []
        predicted = []
        actual = []
        for row in test_rows:
            if row['job'] not in ('plate', 'switch'):
                continue
            guess = frozen.get((row['clue'], row['symbol']))
            if guess is None:
                continue
            mask.append(row)
            predicted.append(guess)
            actual.append(row['job'])
        if actual:
            symbol_accuracy = float(np.mean([guess == truth for guess, truth in zip(predicted, actual)]))
            subset = np.stack([row['features'] for row in mask])
            subset_y = np.array([1.0 if row['job'] == 'plate' else 0.0 for row in mask])
            position_accuracy = accuracy((predict_logistic(weight, mean, std, subset) >= 0.5).astype(np.float64), subset_y)
    clue_map = {}
    clue_counts = {symbol: {0: 0, 1: 0} for symbol in range(MESSAGE_COUNT)}
    for row in validation_rows:
        clue_counts[row['symbol']][row['clue']] += 1
    for symbol, counts in clue_counts.items():
        if counts[0] == counts[1]:
            continue
        clue_map[symbol] = 0 if counts[0] > counts[1] else 1
    clue_hits, clue_total = 0, 0
    for row in test_rows:
        if row['symbol'] not in clue_map:
            continue
        clue_total += 1
        clue_hits += int(clue_map[row['symbol']] == row['clue'])
    stats = dict(
        validation_messages=len(validation_all),
        heldout_messages=len(test_all),
        heldout_labeled=int(sum(row['job'] in ('plate', 'switch') for row in test_rows)),
        position_job_accuracy=position_accuracy,
        symbol_job_accuracy=symbol_accuracy,
        clue_accuracy=None if clue_total == 0 else clue_hits / clue_total,
        clue_predictions=clue_total,
        **intervention,
    )
    conclusion = message_conclusion(stats)
    if not communication:
        conclusion = dict(
            kind='silent', predicts_job=False, changes_partner=False, complement_rate=conclusion['complement_rate'],
            text=('This team was trained with silence delivered to both agents. A one-step symbol injection '
                  'is outside the training distribution and is not evidence of a learned message convention.'),
        )
    payload = dict(communication=bool(communication),
        decision_rule=dict(
            min_interpretation_count=MIN_INTERPRETATION_COUNT, accuracy_margin=INTENT_ACCURACY_MARGIN,
            min_labeled=INTENT_MIN_LABELED, min_branches=INTENT_MIN_BRANCHES,
            complement_rate=INTENT_COMPLEMENT_RATE, clue_accuracy=CLUE_ACCURACY,
        ),
        interpretation=dict(
            labeled_as='validation inference, not an assigned meaning',
            min_count=MIN_INTERPRETATION_COUNT, by_clue=public,
        ),
        stats=stats, conclusion=conclusion,
    )
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'diagnostic.json').write_text(json.dumps(payload, indent=2))
        (out / 'interpretation.json').write_text(json.dumps(payload['interpretation'], indent=2))
    print(json.dumps(dict(conclusion=conclusion, stats=stats), indent=2), flush=True)
    return payload


def smoke(steps=8192):
    """Both channel conditions, real rollout shape, short of the full budget."""
    for communication, name in ((True, 'smoke_comm'), (False, 'smoke_silent')):
        train(ROOT / name, seed=7, steps=steps, communication=communication, periodic_eval=False, final_eval=False)
        model, payload = load_role(ROOT / name / 'checkpoint.pt')
        assert payload['message_count'] == MESSAGE_COUNT
        records = run_cases(model, evaluation_cases(range(21000, 21001)), communication)
        assert len(records) == 8
        (ROOT / name / 'smoke_eval.json').write_text(json.dumps(summarize_episodes(records, 7, communication, steps), indent=2))
    print(json.dumps(dict(smoke_steps=steps, conditions=['communication', 'silent'], episodes_per_condition=8)))


def report(root=ROOT):
    root = Path(root)
    runs = []
    for seed in (7, 8, 9):
        for communication, name in ((True, 'comm'), (False, 'silent')):
            folder = root / f'seed{seed}_{name}'
            if not (folder / 'evaluation.json').exists():
                continue
            item = dict(seed=seed, communication=communication, folder=str(folder))
            item['evaluation'] = json.loads((folder / 'evaluation.json').read_text())
            if (folder / 'paired.json').exists():
                item['paired'] = json.loads((folder / 'paired.json').read_text())
            if (folder / 'diagnostic.json').exists():
                item['diagnostic'] = json.loads((folder / 'diagnostic.json').read_text())
            runs.append(item)
    replay = None
    for item in runs:
        if not item['communication']:
            continue
        showcase = item['folder']
        path = Path(showcase) / 'pair_showcase.json'
        candidate = Path(showcase) / 'replay_candidate.json'
        if path.exists():
            spec = json.loads(path.read_text())
            model, _ = load_role(Path(showcase) / 'checkpoint.pt')
            base = plain_config(random_room(spec['layout_seed']))
            other = exchange_plate_and_switch(base, spec['clue'])
            episodes = []
            for config, note in ((base, 'original plate and correct switch'), (other, 'plate and correct switch exchanged')):
                env = reset_for_evaluation(config, spec['clue'], spec['holder'], spec['swap'], True, spec['layout_seed'])
                actions = play(model, env, case_generator(spec['layout_seed'], spec['clue'], spec['holder'], spec['swap']))
                episodes.append(dict(
                    layout_seed=spec['layout_seed'], clue=spec['clue'], holder=spec['holder'], swap=bool(spec['swap']),
                    config=plain_config(dict(env.config, interchangeable=True)), actions=actions, note=note,
                    outcome=env.outcome, steps=env.steps, unlocked=env.unlocked,
                    plate_holder=None if env.plate_holder is None else int(env.plate_holder),
                    switch_activator=None if env.switch_activator is None else int(env.switch_activator),
                ))
            if episodes[0]['plate_holder'] != episodes[1]['plate_holder'] and episodes[0]['unlocked'] and episodes[1]['unlocked']:
                replay = dict(kind='role pair', checkpoint=str(Path(showcase) / 'checkpoint.pt'), episodes=episodes)
                break
        if replay is None and candidate.exists():
            episode = json.loads(candidate.read_text())
            replay = dict(kind='single episode', checkpoint=str(Path(showcase) / 'checkpoint.pt'), episodes=[episode])
    if replay is not None:
        (root / 'replay.json').write_text(json.dumps(replay))
    # Prefer a communicating interpretation for the launcher.
    for item in runs:
        source = Path(item['folder']) / 'interpretation.json'
        if item['communication'] and source.exists():
            (root / 'interpretation.json').write_text(source.read_text())
            break
    public_runs = []
    for item in runs:
        entry = dict(seed=item['seed'], communication=item['communication'], evaluation={
            key: item['evaluation'][key] for key in (
                'episodes', 'unlocked', 'escaped', 'unlock_rate', 'escape_rate', 'mean_success_steps', 'outcomes')
        })
        if 'paired' in item:
            entry['paired'] = {key: item['paired'][key] for key in (
                'pairs', 'denominator', 'both_solved', 'one_solved', 'neither_solved', 'job_changed',
                'job_changed_and_both_escaped')}
        if 'diagnostic' in item:
            entry['messages'] = dict(conclusion=item['diagnostic']['conclusion'], stats=item['diagnostic']['stats'])
        public_runs.append(entry)
    summary = dict(runs=public_runs, replay=None if replay is None else dict(kind=replay['kind'], episodes=[
        {key: episode[key] for key in episode if key not in ('actions', 'config')} for episode in replay['episodes']
    ]))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def experiment(steps=1_048_576):
    ROOT.mkdir(parents=True, exist_ok=True)
    commands = [
        'python role_ppo.py smoke',
        'python role_ppo.py train --seed 7 --steps 1048576 --out runs/role_coordination/seed7_comm',
        'python role_ppo.py train --seed 7 --steps 1048576 --no-communication --out runs/role_coordination/seed7_silent',
        'python role_ppo.py train --seed 8 --steps 1048576 --out runs/role_coordination/seed8_comm',
        'python role_ppo.py train --seed 8 --steps 1048576 --no-communication --out runs/role_coordination/seed8_silent',
        'python role_ppo.py train --seed 9 --steps 1048576 --out runs/role_coordination/seed9_comm',
        'python role_ppo.py train --seed 9 --steps 1048576 --no-communication --out runs/role_coordination/seed9_silent',
        'python role_ppo.py report',
        'python demo.py --controller roles --checkpoint runs/role_coordination/seed7_comm/checkpoint.pt --replay runs/role_coordination/replay.json',
    ]
    (ROOT / 'commands.txt').write_text('\n'.join(commands) + '\n')
    smoke()
    for seed in (7, 8, 9):
        for communication, name in ((True, 'comm'), (False, 'silent')):
            folder = ROOT / f'seed{seed}_{name}'
            train(folder, seed=seed, steps=steps, communication=communication)
            model, _ = load_role(folder / 'checkpoint.pt')
            paired_roles(model, communication, folder)
            diagnose(model, communication, folder)
    return report(ROOT)


def main():
    parser = argparse.ArgumentParser(description='Train and test role coordination in the sunken vault')
    sub = parser.add_subparsers(dest='command', required=True)
    smoke_parser = sub.add_parser('smoke')
    smoke_parser.add_argument('--steps', type=int, default=8192)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--seed', type=int, required=True)
    train_parser.add_argument('--steps', type=int, default=1_048_576)
    train_parser.add_argument('--out', required=True)
    train_parser.add_argument('--no-communication', action='store_true')
    evaluate_parser = sub.add_parser('evaluate')
    evaluate_parser.add_argument('--checkpoint', required=True)
    paired_parser = sub.add_parser('paired')
    paired_parser.add_argument('--checkpoint', required=True)
    diagnose_parser = sub.add_parser('diagnose')
    diagnose_parser.add_argument('--checkpoint', required=True)
    sub.add_parser('report')
    experiment_parser = sub.add_parser('experiment')
    experiment_parser.add_argument('--steps', type=int, default=1_048_576)
    args = parser.parse_args()
    if args.command == 'smoke':
        smoke(args.steps)
    elif args.command == 'train':
        train(args.out, seed=args.seed, steps=args.steps, communication=not args.no_communication)
    elif args.command == 'evaluate':
        model, payload = load_role(args.checkpoint)
        evaluate(model, payload['seed'], payload['communication'], payload['steps'], Path(args.checkpoint).parent)
    elif args.command == 'paired':
        model, payload = load_role(args.checkpoint)
        paired_roles(model, payload['communication'], Path(args.checkpoint).parent)
    elif args.command == 'diagnose':
        model, payload = load_role(args.checkpoint)
        diagnose(model, payload['communication'], Path(args.checkpoint).parent)
    elif args.command == 'report':
        report()
    else:
        experiment(args.steps)


if __name__ == '__main__':
    main()
