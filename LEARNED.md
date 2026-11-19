# Learned cooperation in the escape room

Double-click **Watch Learned Cooperation.cmd**. It loads seed 7's trained policies. Each new round uses a different room. The message panel shows what A sends, and both agents move automatically.

## What the agents learned

A selects a symbol from two anonymous alternatives after seeing the private clue. With messaging enabled, B receives it on the next step. Each agent independently chooses a destination from the plate, either switch, or exit. Agents choose again when the door opens. The training code supplies the decision times, available destinations, pathfinding, and interaction upon reaching a switch. Neither agent receives a target destination or a correct message label.

The two agents train from scratch with independent tabular policy gradients and a common episode return. Training uses 500 batches of 128 joint decisions on the original fixed room. A cache reuses deterministic room outcomes for repeated goal combinations. A's locked-phase goal and symbol depend on the clue, B's locked-phase goal depends on its received symbol, and each agent learns its own exit-phase goal. There is no scripted solution in the policy update. Policies choose the highest-scoring actions during evaluation. This is a separate method from the earlier primitive-action PPO experiments.

## Frozen policy evaluation

The same 100 held-out layouts (seeds 17000–17099) were used for each condition. Each layout was evaluated with both private clues, giving 200 episodes. Layout generation admits rooms the reference controller can solve under both clues. The 200 episodes are paired by layout, and all teams share the same evaluation layouts.

| Training seed | Messaging team | Same team, symbols swapped | Same team, silenced | Team trained without channel | Rooms solved under both clues with messaging |
|---|---:|---:|---:|---:|---:|
| 7 | 179/200 | 0/200 | 0/200 | 93/200 | 85/100 |
| 8 | 179/200 | 0/200 | 93/200 | 1/200 | 85/100 |
| 9 | 179/200 | 0/200 | 93/200 | 93/200 | 85/100 |

Swapping symbols preserves the two inputs B encountered during training but reverses their relationship to A's clue. The collapse in success supports reliance on learned message meaning. Silencing uses a receiver input the communicating team never trained on, so the swap is the cleaner intervention. None of the three silent-baseline teams solved both clue variants in any layout. The seed 7 messaging team's remaining 21 episodes timed out.

These results apply to agents with destination choices and a supplied navigator. They establish learned coordination within that design. They do not establish end-to-end learning of movement or the superiority of explicit messaging for agents free to learn other signaling mechanisms.

## Reproduce

```powershell
python options_rl.py --out runs/options_reproduction
python demo.py --controller options --checkpoint runs/options_reproduction/seed7_comm.json
```

The training command creates six policy checkpoints and a machine-readable results table. Pass `--seed 17000` to the viewer to start at a reproducible layout. No additional packages or model downloads are required.
