# Partner handoff · Cooperative Escape Room

We started with one question: can two learning agents use a private clue to escape together? We now have a reproducible demonstration of learned communication when navigation is supplied. The more ambitious version gives both agents the whole public map, lets either take either role, and requires them to learn every move. It can occasionally unlock an unfamiliar room, but it has yet to escape one in the recorded evaluation.

## First session together

1. Follow the [README setup](README.md). Watch **Play Demo.cmd** once to learn the puzzle, then **Watch Learned Cooperation.cmd** to see the successful learned-message controller. The latter chooses destinations and uses a supplied pathfinder.
2. Open **Watch Learned Moves.cmd** to see self-selected physical roles on the familiar map. Press **R** to try another clue or starting-side assignment. Open **Watch Full Map Agents.cmd** for the full-map pilot. Open **Watch Role Coordination.cmd** for the shared GRU policy with no assigned jobs. The account of that run is in [Role coordination](#role-coordination-what-this-branch-contains).
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

## Role coordination: what this branch contains

This is the work on `role-coordination`. The full-map pilot remains the baseline: shared weights, interchangeable mechanics, primitive actions, and **0/100** escapes. The **179/200** figure belongs to the destination controller that uses a supplied pathfinder. The four-page paper was left for a later pass.

### Code that was added

| Piece | What it does |
|---|---|
| [role_ppo.py](role_ppo.py) | The experiment. One shared policy. Each agent has its own 64-unit GRU. A convolutional encoder reads seven 9×7 layers: walls, plate, both switches, exit, self, and partner. Those features are combined with the private clue, the partner's previous message, the door, the escape flags, and the time remaining. |
| [environment.py](environment.py) | The room can carry a message width and a movement table. The old controllers still use three messages and six movements. This experiment uses silence plus four anonymous symbols. Nothing in the code names those symbols. |
| [demo.py](demo.py) and [vault_view.py](vault_view.py) | Controller `roles`. The panel shows the raw symbol. After an unlock it names who held the plate and who activated the switch. A symbol gloss is labeled as a validation inference. `--campaign` and `--relay` play the two-stage vault. |
| [relay_room.py](relay_room.py), [Watch Two Stages.cmd](Watch%20Two%20Stages.cmd), [Play Relay Chamber.cmd](Play%20Relay%20Chamber.cmd) | After the vault, the scripted crew hands a fuse across a grille. Manual play can start in that chamber. Role checkpoints stay on the six-action vault. |
| [Watch Role Coordination.cmd](Watch%20Role%20Coordination.cmd) | Windows launcher for the seed 7 messaging checkpoint. |
| [test_role_ppo.py](test_role_ppo.py) | Regression tests, listed below. |

Either agent can hold the plate while the other activates the correct switch. The clue value, the clue recipient, and which agent stands on which start tile are randomized separately. A and B exist in the viewer and the logs. The policy never receives an agent id, a role label, a destination, a path, or an expert action. Memory resets when the episode ends. The PPO update replays each rollout as a contiguous sequence so the GRU is trained through time. With the two agents' inputs and memories swapped, the shared policy returns the swapped action distributions.

The silent team uses the same network size. The room delivers silence to both agents on every step.

Training practice follows the budget. The team reward is the original room reward throughout.

| Budget | Practice |
|---|---|
| First 20% | The exit starts open. The agents start on two distinct reachable floor tiles. The team gets +1 once, when the first agent exits. |
| Next 30% | The door is locked. One start is near the plate and one is near a switch. Which agent gets which start, and who sees the clue, is random. Neither agent starts on a mechanism. The same one-time exit bonus applies. |
| Last 50%, and every reported evaluation | The full locked puzzle, ordinary starts, private clue, and the original rewards only. |

PPO settings match the repository baseline: 16 environments, 128-step rollouts, learning rate 0.0003, Adam epsilon 1e-5, discount 0.99, GAE lambda 0.95, clip 0.2, entropy coefficient 0.01, four epochs, four minibatches, gradient clip 0.5. Training layouts are seeds 0–1023, cached in [train_layouts.json](runs/role_coordination/train_layouts.json). Validation layouts are seeds 20000–20024 and were used for the periodic check and for freezing symbol readings. Held-out layouts are seeds 21000–21099. Each layout is run under both clues, both recipients, and both start assignments: 800 episodes per finished seed and condition. The same cases are used for both channel conditions.

The paired-role test keeps walls, exit, starts, clue, and clue recipient fixed, and exchanges the plate with the correct switch. Both variants have to be solvable by the scripted reference. The denominator is every pair, including failures. Fifty pairs were built from layout seeds beginning at 22000.

The message diagnostic freezes a symbol reading on validation before it looks at held-out episodes. A symbol counts as an intent signal only if that reading predicts the sender's job beyond the clue and the positions, and changing the delivered symbol changes the partner's complementary job. The cutoffs live in `role_ppo.py` and were fixed before the held-out test: at least 30 labeled held-out rows, a 5 point accuracy margin, at least 20 job-contrast branches, and a complementary-change rate of at least 0.20. A success that becomes a failure counts as disruption, not a role change. A message chosen before the first step onto the plate or a switch can still be scored. After that contact, later messages stay ineligible even if both agents walk off.

### Tests that were run

`python -m unittest -v` passed, including the older room, map, and learning tests. The new tests check:

- Either agent can hold the plate while the other activates the correct switch, and the log records those two roles.
- The clue changes only the recipient's observation.
- A symbol sent on one step appears in the partner's next observation.
- With the channel off, both agents receive silence even if they select a symbol. The message head is still width 5.
- Curriculum starts obey the stage rules. Evaluation always begins in the locked full puzzle, and the exit bonus is absent there.
- Exchanging the plate and the correct switch leaves the other switch, the walls, the exit, and the starts in place.
- Resetting one agent's memory leaves the other agent's memory intact.
- Swapping the two agents' inputs and memories swaps the action distributions.
- The GRU receives gradient through time, including across a mid-sequence reset.
- A batched evaluation of a frozen policy matches a solo replay of the same episode.
- A short PPO update changes the policy weights.

A headless viewer frame of controller `roles` showed the raw symbol line, the private-clue line, and the note that symbol readings are validation inferences.

### Training that finished

Smoke test, both channels, 8,192 joint steps, seed 7. Checkpoints are in [smoke_comm](runs/role_coordination/smoke_comm) and [smoke_silent](runs/role_coordination/smoke_silent). The 8-episode smoke evaluation timed out on every episode. It checks that the pipeline runs.

Seed 7 then trained for 1,048,576 joint steps in each condition. Held-out evaluation is 800 episodes. Sources: [seed7_comm/evaluation.json](runs/role_coordination/seed7_comm/evaluation.json), [seed7_silent/evaluation.json](runs/role_coordination/seed7_silent/evaluation.json), [seed7_comm/paired.json](runs/role_coordination/seed7_comm/paired.json), [seed7_silent/paired.json](runs/role_coordination/seed7_silent/paired.json), [seed7_comm/diagnostic.json](runs/role_coordination/seed7_comm/diagnostic.json), [seed7_silent/diagnostic.json](runs/role_coordination/seed7_silent/diagnostic.json).

| Team | Unlocked | Escaped | Successful steps | Other outcomes | Both pair variants escaped | Plate holder changed across a pair |
|---|---:|---:|---|---|---:|---:|
| Seed 7, messages | 14/800 | 1/800 | 91 | 780 timeouts, 19 wrong switches | 0/50 | 1/50 |
| Seed 7, silence | 17/800 | 3/800 | 84 on average (62, 92, and 98) | 778 timeouts, 19 wrong switches | 0/50 | 0/50 |

The messaging success is layout 21007, clue 0, recipient A, starts swapped. B held the plate and A activated the switch. The silent successes are layouts 21013 (92 steps, plate B, switch A), 21036 (62 steps, plate B, switch A), and 21050 (98 steps, plate A, switch B).

The one plate-holder change is layout 22028, clue 1, recipient B, starts swapped. Both variants unlocked and then timed out at step 100. In one variant B held the plate and A activated the switch. In the exchanged room those jobs were reversed. That is one observed job change. It is not a pair of escapes.

On the messaging team the held-out diagnostic had 28 labeled sender-job rows, under the pre-registered minimum of 30. No frozen symbol reading was scored. Clue-from-symbol accuracy was 0.50125 on 1,600 predictions. Of 800 message interventions, the partner's next moves changed 13 times. Recorded role changes were 0, success disruptions were 0, and branches that contrasted two different job readings were 0. The saved conclusion is that there are too few labeled episodes to test intent. No intent signal is claimed. There is also no evidence here that the symbols carry the clue.

On the silent team the room delivered only silence, so the diagnostic had no alternate symbol to substitute (0 eligible branches). Position predicted the sender's job on 22 of 34 labeled rows. That baseline does not use a message. The silent run is the no-channel comparison, not a test of a learned code.

The diagnostic counts above were computed before the first-contact fix; corrected counts pending. The escape counts in the table are unchanged.

### What this branch does not yet have

Seed 8 with messages was still in the near-mechanism stage of its 1,048,576-step run when this account was written. [seed8_comm/training.json](runs/role_coordination/seed8_comm/training.json) is a partial learning curve from that run. It has no checkpoint and no evaluation. Seed 8 without messages, and both seed 9 runs, had not been started. `summary.json` and `replay.json` are written at the end of `python role_ppo.py experiment`, so they are absent. The launcher still opens the finished seed 7 messaging checkpoint and plays it live.

```powershell
python -m unittest -v
python role_ppo.py smoke
python role_ppo.py train --seed 7 --steps 1048576 --out runs/role_coordination/seed7_comm
python role_ppo.py train --seed 7 --steps 1048576 --no-communication --out runs/role_coordination/seed7_silent
python demo.py --controller roles --checkpoint runs/role_coordination/seed7_comm/checkpoint.pt
```

The two `train` commands repeat the finished seed 7 runs. Their checkpoints and evaluations are already in the folders above.

## Course deliverables and ownership

- **Working code:** source, launchers, trained policies, evaluation records, and reproduction commands are in the repository. Baseline results are committed. Verify installation and playback on the partner's machine before presenting.
- **Four-page paper:** still to write in LaTeX using an official major AI conference style. References are outside the four-page limit. Cover motivation, related work, methods, evaluation and limitations. The learned-goal symbol-swap experiment is ready to anchor the results section even if further full-map training is inconclusive.
- **Five-minute presentation and demo:** still to prepare and rehearse. Choose one presenter. Show what information each agent receives, demonstrate one labeled learned controller, and explain a controlled comparison. Both partners should be ready for questions.
- **Coordination:** confirm presentation and submission dates, the final roster, and any approval needed for a team smaller than the assigned four. No dates or individual assignments are recorded here. A practical split is for one partner to own training and evaluation logs and the other to independently reproduce a result and draft the methods and figures. Review conclusions together.

