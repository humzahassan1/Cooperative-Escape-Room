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

## Next experiment to run

The implementation target is a team that sees the room, takes individual steps, chooses its own roles, and escapes on new layouts. Keep the current full-map checkpoint as the baseline. A suggested next run is to change **one training factor**: compare the current starts with staged practice that begins near the exit, then near the plate and switches, then in the full room. Use the same total budget, training layouts, rewards, architecture and matched seeds for both conditions. Save new runs in separate directories and record the exact configuration.

Log intermediate outcomes to identify where training breaks down: plate occupied during a switch attempt, correct unlock, first agent out, both out. Evaluate frozen policies on balanced clue and recipient combinations. Reserve fresh layout seeds for final evaluation once the design is chosen. If escapes become reliable, compare normal messages, swapped messages, and a separately trained silent baseline. Repeat over multiple training seeds. This is proposed work; the repository currently contains one full-map training seed.

## Course deliverables and ownership

- **Working code:** source, launchers, trained policies, evaluation records, and reproduction commands are in the repository. Baseline results are committed. Verify installation and playback on the partner's machine before presenting.
- **Four-page paper:** still to write in LaTeX using an official major AI conference style. References are outside the four-page limit. Cover motivation, related work, methods, evaluation and limitations. The learned-goal symbol-swap experiment is ready to anchor the results section even if further full-map training is inconclusive.
- **Five-minute presentation and demo:** still to prepare and rehearse. Choose one presenter. Show what information each agent receives, demonstrate one labeled learned controller, and explain a controlled comparison. Both partners should be ready for questions.
- **Coordination:** confirm presentation and submission dates, the final roster, and any approval needed for a team smaller than the assigned four. No dates or individual assignments are recorded here. A practical split is for one partner to own training and evaluation logs and the other to independently reproduce a result and draft the methods and figures. Review conclusions together.

