# Partner handoff · Cooperative Escape Room

We started with one question: can two learning agents use a private clue to escape together? We now have a reproducible demonstration of learned communication when navigation is supplied. The more ambitious version gives both agents the whole public map, lets either take either role, and requires them to learn every move. It can occasionally unlock an unfamiliar room, but it has yet to escape one in the recorded evaluation.

## First session together

1. Follow the [README setup](README.md). Watch **Play Demo.cmd** once to learn the puzzle, then **Watch Learned Cooperation.cmd** to see the successful learned-message controller. The latter chooses destinations and uses a supplied pathfinder.
2. Open **Watch Learned Moves.cmd** to see self-selected physical roles on the familiar map. Press **R** to try another clue or starting-side assignment. Open **Watch Full Map Agents.cmd** to see the version we want to improve.
3. Run `python -m unittest -v`. Inspect the [goal-policy results](runs/options/results.json), [role and transfer summary](runs/free_roles/summary.json), and [full-map pilot](runs/map_ppo/evaluation.json). These files are the sources for the numbers below.

## How we got here

| Decision | Reason and observed result |
|---|---|
| Keep exactly two agents and one 9×7 room. Use Python, NumPy, PyTorch and Pygame. | The same headless simulation runs training and the visual demo. A plate, two switches and a private clue make it possible to test whether the agents use information from a partner. A scripted pair solved 200/200 fixed-room trials in 11 steps on average. |
| Start with independent PPO, fixed roles and identical training budgets. | Each agent chose a move and a symbol and received the team reward. After 501,760 joint steps per condition, both the messaging team and the separately trained silent team escaped 0/200 times. [Pilot](RESULTS.md) |
| Try staged practice and progress rewards. | At 131,072 steps, the messaging team escaped 12/200, the silent team 25/200, and the messaging team with delivery silenced 11/200. Neither trained team chose correctly under both clues in any of 100 matched pairs. This single-seed pilot changed both the training tasks and the rewards. [Staged experiment](STAGED.md) |
| Isolate communication by supplying movement between destinations. | Independent tabular policy gradients learned symbol and goal choices. Three seeds each achieved 179/200 on 100 unfamiliar layouts tested under both clues. Each solved both clues on 85/100 layouts. Reversing the symbols at evaluation produced 0/200 every time. [Learned goals](LEARNED.md) |
| Remove pathfinding and fixed physical roles. | A shared table chose primitive moves. An evolutionary strategy trained it using team performance and an exit-distance bonus after unlocking. All three seeds solved all eight fixed-room combinations of starting assignment, clue holder and clue value. Swapping IDs switched who held the plate. Each policy timed out on all 100 unfamiliar-layout trials. Its inputs omitted walls and object positions. [Interchangeable roles](FREE_ROLES.md) |
| Give both agents full public map visibility. | Shared PPO now uses seven board layers for walls, plate, both switches, exit, self and partner. The private clue goes only to its recipient. Both agents choose primitive moves and messages. Seed 7 trained for 65,536 joint steps across 128 layouts. On 25 held-out layouts × two clues × two clue holders it unlocked 3/100 and escaped 0/100. [Implementation](map_ppo.py) · [Results](runs/map_ppo/evaluation.json) |

Randomized layouts are filtered for solvability by replaying the scripted reference under both clues. That check happens during layout generation. The primitive-action policies receive no route from it. The project implements established algorithms for this environment: PPO, tabular policy gradients, and evolutionary search. Our contribution is the environment, controlled experiments, and analysis.

## What the evidence supports

**Communication.** In the learned-goal experiment the weights, maps, and clues stayed fixed while the evaluator swapped the symbols. Success fell from 179/200 to 0/200 at all three seeds. This supports a learned signal convention within that setup. The separately trained silent teams escaped 93, 1, and 93 times out of 200 respectively. Keep the independent training seeds distinct from the 200 evaluation episodes.

**Role selection.** The evolved policy used the same weights for both agents. The demo team learned a starting-side convention. Either identity could operate the plate. We have evidence of role selection through position; explicit negotiation was never demonstrated.

**Generalization.** The 179/200 result belongs to the version with built-in navigation. Full-map PPO uses a different policy, action set, reward setup, and evaluation design. Comparing those percentages directly cannot isolate the effect of giving agents a map. A message intervention on a team with zero escapes also provides little information about what its messages mean.

## Role coordination

The full-map pilot is the baseline this run starts from: shared weights, interchangeable mechanics, primitive actions, and **0/100** escapes. The **179/200** result is a different controller that chooses destinations and uses a supplied pathfinder. It is not the comparison for this run.

`role_ppo.py` keeps one shared policy. Each agent has its own 64-unit GRU memory. A small convolutional encoder reads the 9×7 board (walls, plate, both switches, exit, self, partner). That vector is combined with the private clue, the partner's previous message, the door, escape flags, and the remaining time. Memory resets at the episode boundary, and the PPO update replays contiguous rollouts so the GRU is trained through time. Both agents may send silence or one of four anonymous symbols. The code does not assign those symbols a meaning. The silent team uses the same network and receives silence on every step.

Practice changes with the training budget, and the team reward stays the original room reward. For the first 20% the exit starts open and the agents start on distinct reachable tiles, with a one-time bonus of +1 when the first agent exits. For the next 30% the door is locked and the starts are one tile near the plate and one near a switch, assigned at random, with the same bonus. For the last 50%, and for every reported evaluation, the episode is the full locked puzzle and the bonus is absent. No reward depends on who takes a job or which symbol is sent.

The smoke test is 8,192 joint steps for both channel conditions. The recorded run is seeds 7, 8, and 9, each for 1,048,576 joint steps with and without the channel. Rollouts are 16 environments by 128 steps. The other PPO settings match the repository baseline: learning rate 0.0003, Adam epsilon 1e-5, discount 0.99, GAE lambda 0.95, clip 0.2, entropy 0.01, four epochs, four minibatches, and gradient clip 0.5. Training layouts are seeds 0–1023. Validation layouts are seeds 20000–20024. Held-out layouts are seeds 21000–21099, eight episodes each (both clues, both recipients, both start assignments), 800 episodes per seed and condition.

A second test builds paired rooms that keep the walls, exit, starts, clue, and clue recipient fixed, and exchange the plate with the correct switch. Both variants are checked with the scripted reference. The denominator includes pairs where one or both attempts fail.

The message diagnostic freezes any symbol reading on validation, before held-out episodes. A symbol is described as an intent signal only if that frozen reading predicts the sender's job beyond the clue and positions, and replacing the symbol changes the partner's complementary job. The cutoffs are fixed in `role_ppo.py`: at least 30 labeled held-out rows, a 5 point accuracy margin, at least 20 job-contrast branches, and a complementary-change rate of 0.20. A crash from success to failure counts as disruption, not a role change. If symbols predict the clue and not the job, the report says clue transmission.

```powershell
python role_ppo.py smoke
python role_ppo.py experiment
python demo.py --controller roles --checkpoint runs/role_coordination/seed7_comm/checkpoint.pt --replay runs/role_coordination/replay.json
```

Per-seed escape rates, paired-role counts, and the message conclusion are in [summary.json](runs/role_coordination/summary.json). `Watch Role Coordination.cmd` replays the saved episodes, shows the raw symbols, and names the plate holder and switch activator after an unlock. Any symbol gloss on screen is labeled as a validation inference.

## Course deliverables and ownership

- **Working code:** source, launchers, trained policies, evaluation records, and reproduction commands are in the repository. Baseline results are committed. Verify installation and playback on the partner's machine before presenting.
- **Four-page paper:** still to write in LaTeX using an official major AI conference style. References are outside the four-page limit. Cover motivation, related work, methods, evaluation and limitations. The learned-goal symbol-swap experiment is ready to anchor the results section even if further full-map training is inconclusive.
- **Five-minute presentation and demo:** still to prepare and rehearse. Choose one presenter. Show what information each agent receives, demonstrate one labeled learned controller, and explain a controlled comparison. Both partners should be ready for questions.
- **Coordination:** confirm presentation and submission dates, the final roster, and any approval needed for a team smaller than the assigned four. No dates or individual assignments are recorded here. A practical split is for one partner to own training and evaluation logs and the other to independently reproduce a result and draft the methods and figures. Review conclusions together.

