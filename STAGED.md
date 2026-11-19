# Testing whether the agents learn to cooperate

Success requires A to hold the plate, B to activate the correct switch, and both agents to escape. The evaluator runs frozen policies without giving either agent scripted actions. A alone sees the clue. Both agents see positions and can potentially signal through movement.

## Reproduce the staged pilot

Run from the project directory. Both conditions use the same seed, budget and training rewards.

```powershell
python learning.py train --staged --steps 131072 --out runs/staged_comm
python learning.py train --staged --steps 131072 --no-communication --out runs/staged_silent
python cooperation_check.py runs/staged_comm/checkpoint.pt
python cooperation_check.py runs/staged_silent/checkpoint.pt
```

The first quarter of training starts with an open exit and both agents nearby. For the next 30% of the budget A starts on the plate. Later episodes start in the original room. During training a shared bonus rewards net progress toward the plate and either switch before unlocking, then progress toward the exit. The bonus does not identify the correct switch. These changes are a combined intervention. The shorter training budget also differs from the original sparse-reward pilot.

## Read the evidence

- **Learning:** Compare initial and final escape counts in each run's `paired_clues.json`. The weights change only during training. Evaluation always starts in the original locked room with the original rewards.
- **Conditional cooperation:** The paired test flips only A's clue and matches the action-sampling randomness across both trials. Count how often B activates the correct switch under both clues and how often both agents escape under both clues. A strategy that always chooses one switch cannot pass both members of a pair.
- **Explicit communication:** Compare final success in `evaluation.json` with the same trained weights under `evaluation_silence.json` and `evaluation_random.json`. Compare also with the separately trained silent team. A drop under message disruption is evidence of channel dependence. Movement remains visible, and changing message inputs can also disrupt a policy for reasons unrelated to clue meaning.
- **Replication:** Repeat the matched training and tests with multiple independent training seeds. The 200 evaluation episodes for a single trained pair measure that pair's behavior. They do not measure variation across training runs. Claims about unfamiliar layouts require evaluating those layouts separately.

## Single-seed results

- Messaging escaped 12/200 episodes.
- No channel during training escaped 25/200 episodes.
- Messaging team silenced escaped 11/200 episodes.
- Messaging team with random symbols escaped 10/200 episodes.

Neither trained team activated the correct switch on both members of any of 100 matched clue pairs. These runs occasionally solve the puzzle. Reliable clue-dependent cooperation remains unestablished.
