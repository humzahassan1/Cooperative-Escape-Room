# Cooperative Escape Room MVP

**Full-map pilot:** open **Watch Full Map Agents.cmd**. Both agents receive the complete public board each turn. Seven 7×9 layers identify walls, plate, numbered switches, exit, self and partner. Each also receives lock state, escape flags, remaining time, its own private clue field and the partner's previous message. One shared PPO policy learns primitive moves and signals on 128 randomized training layouts. Both agents can perform either puzzle action. To reproduce the 65,536-step CPU pilot and its evaluation on 25 held-out layouts, run `python map_ppo.py`. Evaluation counts both clues and both possible clue holders per layout. Results are saved in `runs/map_ppo/evaluation.json`.

**To watch role selection and individual moves learned on the fixed room**, open **Watch Learned Moves.cmd**. Either agent can use either mechanism. Read [the method and transfer results](FREE_ROLES.md).

**Working learned demo:** open **Watch Learned Cooperation.cmd**. Independently trained agents choose messages and destinations and use a supplied pathfinder. They escaped in 179/200 held-out trials. See [the learned-cooperation experiment](LEARNED.md) for the controls, results, and reproduction commands. The PPO pilot below documents the earlier method.

A two-agent grid escape room with a Pygame viewer, decentralized scripted reference, and independent PPO in PyTorch. **The first trained policies did not learn to solve the room.** See `RESULTS.md` for the actual pilot results. Scripted mode is explicitly labeled and demonstrates the intended mechanics.

## Setup

Python 3.12 recommended. From this directory, create an isolated environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Use `.venv\Scripts\python` instead of `python` below if the environment is not activated. CPU is sufficient. No external datasets, pretrained models, accounts, or services are needed. `tested_environment.json` records the original run environment. Pygame-ce is the recommended renderer; the initial preview was also checked with the host's legacy Pygame 2.5.2.

## Play and watch

On this Windows machine, double-click `Play Demo.cmd` to watch the scripted solution or `Play Manual.cmd` to play. These use the project virtual environment if present, otherwise the installed Python.

```powershell
python demo.py --controller manual
python demo.py --controller scripted
python demo.py --controller ppo --checkpoint runs/communication/checkpoint.pt
python demo.py --controller ppo --checkpoint runs/no_communication/checkpoint.pt
```

Controls: **Tab** selects an agent; **arrows** move; **E** interacts with a switch underfoot; **Space** waits; **0/1/2** selects the outgoing symbol. Choosing a symbol alone does not advance time. Each movement/interact/wait advances one joint step, with the unselected agent waiting and sending its selected symbol. **R** starts the next round, **P** pauses, **N** single-steps automatic controllers, **+/-** adjusts playback speed, and **Esc** quits. Manual and scripted play generate new walls, starts, plate, switches, and exit each round. Each layout is checked with both clues. Automatic controllers start another round after two seconds. Use `--seed 43` to reproduce a sequence. PPO replay uses the original training layout. The observer panel reveals A's clue to the human, but B's policy never receives it.

A must stand on the plate while B interacts with the switch matching A's clue. A wrong active switch ends the episode. Unlocking is permanent; both agents then reach the exit, one at a time. Agents that exit leave the board. Exactly two agents are supported.

## Reproduce training and evaluation

Use new output directories to preserve the delivered pilot:

```powershell
python learning.py train --out runs/reproduction_communication --seed 7 --steps 500000
python learning.py train --out runs/reproduction_no_communication --seed 7 --steps 500000 --no-communication
python learning.py evaluate --checkpoint runs/communication/checkpoint.pt --out runs/rechecked.json
python learning.py evaluate --checkpoint runs/communication/checkpoint.pt --mode silence --out runs/rechecked_silence.json
python learning.py evaluate --checkpoint runs/communication/checkpoint.pt --mode random --out runs/rechecked_random.json
python report.py
python -m unittest -v
```

`report.py` reads the delivered `runs/communication` and `runs/no_communication` results, re-evaluates random/scripted references, and recreates `RESULTS.md` and `results.png`. Training rounds the requested budget up to a complete 2,048-step rollout: 500,000 becomes 501,760 joint steps. Each joint step advances both agents and is counted once. Checkpoints contain both policies, room config, observation version, and step count. They support evaluation/replay, not optimizer-state resume. Never load untrusted PyTorch checkpoints.

Periodic evaluation uses 40 balanced-clue episodes at fixed seeds 5000–5039, after update 1, every 25 updates, and at the end. Final evaluation uses 200 episodes at seeds 10000–10199, with alternating clues. Policies sample actions during evaluation, matching training behavior; seeds are fixed for reproducibility. Numerical reproducibility can vary with PyTorch/platform versions. No best-checkpoint selection is performed. `runs/smoke` preserves a one-update development check, not a competing experiment.

## Structure and extension points

- `environment.py`: NumPy-only joint-step simulation and observation-only scripted agents.
- `room.json`: dimensions, fixed starts, plate, switches, exit, episode limit.
- `learning.py`: separate actor-critic networks/optimizers, PPO training, checkpoint loading, evaluation.
- `demo.py`: rendering and human/automatic playback, independent from training.
- `report.py`, `test_environment.py`, `test_learning.py`: analysis and verification.

API: `reset(seed, clue=None)` returns `(observations_by_agent, info)`. `step({'A': (action, message), 'B': (...)})` returns `(observations, shared_rewards_by_agent, terminated, truncated, info)`. Agent IDs are `A` and `B`. Actions are up=0, down=1, left=2, right=3, wait=4, interact=5. Symbols are 0, 1, 2; only the scripted controller assigns meaning to them. Delivered symbols become visible in the next observation. No-communication runs replace every delivered symbol with 0.

Observation v1 has 21 floats: normalized A/B positions (4), plate/switch1/switch2/exit positions (8), unlocked (1), escaped A/B (2), remaining fraction (1), private clue one-hot (2; zero for B), and partner message one-hot (3). Both conditions use the same architecture. Each policy has a shared 64×64 tanh trunk and separate movement, message, and value heads. Both agents have independent weights and optimizers. A value target bootstraps at timeouts, but GAE does not cross resets; true terminations do not bootstrap.

Positions remain recorded after escape, paired with explicit escaped flags. Moves resolve together: shared destinations, swaps, and entering a stationary agent's cell are blocked. Following a moving agent is allowed. The exit opens after movement and interaction, so entry begins on a later step.

The simulation supports configurable coordinates; the MVP renderer and scripted pathfinding assume a 9×7 board. New board sizes or object types need renderer/script updates and observation-version checks. Checkpoint replay verifies the saved room matches the current room. Changing mechanics requires retraining and separately named result directories.

## Verification and limitations

All 12 automated tests pass. Environment tests cover reset determinism, privacy, messages, collisions, coordinated unlocking, rewards, failure, exit, timeouts, and 200 successful scripted episodes. Learning checks cover finite updates, changed parameters, checkpoint roundtrips, and timeout versus terminal targets. The live scripted window and readable layout were visually checked; keyboard automation was unreliable, so pause/restart interaction still warrants a human check. Installing the recommended Pygame-ce could not be verified because package-registry DNS was unavailable; the installed Pygame 2.5.2 successfully ran the demo.

This is a single-seed pilot, not the final paper. Both learned conditions failed; successful scripted behavior is not evidence of learned cooperation. No extra shaping, imitation learning, curriculum, or hidden policy access to the clue was used.
