# Learning moves with interchangeable agents

Open **Watch Learned Moves.cmd**. Each round changes the clue, its recipient, and/or the assignment of agents to the two starting tiles. The room geometry is fixed.

Either agent can hold the plate while the other interacts with the matching switch. Each tick an agent chooses up, down, left, right, wait, or interact. Its policy is a table indexed by its own tile, the door state, and its private clue or received symbol. The clue recipient broadcasts a learned symbol. The partner sends silence. Both agents use identical weights. Neither receives a named job, route, or destination. A physical role can emerge from position even with no explicit negotiation.

Training uses an established (1 + 255) evolutionary strategy with a discrete mutation operator. One candidate table controls both agents. Each generation evaluates the current table and 255 variants over both clues and both clue recipients. It keeps a best-scoring table with random tie breaking. This is joint policy optimization with decentralized execution. Principles of the algorithm are described by [Beyer and Schwefel](https://doi.org/10.1023/A:1015059928466).

Training fitness includes the shared room rewards and a bonus of 0.12 times the improvement from a reference combined Manhattan distance of 24 to the closest combined exit distance after unlocking. The training horizon is 32 moves, with a timeout step cost for the unused remainder of the 100-step room budget. Evaluation uses the actual room with its 100-step limit and original rewards. This bonus supplies feedback about approaching the exit without selecting moves or assigning roles.

| Training seed | Fixed room, all 8 combinations | Same policy silenced | Unseen layouts |
|---|---:|---:|---:|
| 7 | 8/8 | 6/8 | 0/100 |
| 8 | 8/8 | 6/8 | 0/100 |
| 9 | 8/8 | 6/8 | 0/100 |

The eight deterministic combinations cross starting-side assignment, clue recipient and clue value. They are exhaustive checks on the fixed puzzle. Swapping IDs tests symmetry. Each ID held the plate in four trials per policy. The demo policy learned a starting-side convention. Its table omits wall and object locations, which limits transfer. On 25 new solvable layouts with both clue values and recipients, all three policies timed out. The silencing comparison tests the frozen policy's dependence on messages. A separately trained no-channel baseline would be needed to compare learning with and without the channel.

## Reproduce

```powershell
python free_roles.py
python check_free_roles.py
python demo.py --controller free
```

Training writes three checkpoints with training traces and individual episode outcomes to the free_roles run directory. The evaluation script replays them in the game and writes summary.json. The earlier PPO and goal-selection experiments use different rules and policy structures.
