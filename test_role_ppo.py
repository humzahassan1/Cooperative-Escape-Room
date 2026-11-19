import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from environment import EscapeRoom, IDS
from role_ppo import (
    BOARD, EXTRAS, HIDDEN, MESSAGE_COUNT, OBS_SIZE, RolePolicy, RoleRoom,
    apply_done, case_generator, evaluation_cases, exchange_plate_and_switch,
    mechanisms, play, reset_for_evaluation, run_cases,
    sample_exit_starts, sample_near_starts, sequence_forward, shaped_reward,
    stage_for, stream_rngs,
)


def fixed_config():
    return dict(EscapeRoom().config, interchangeable=True)


class RoleRoomTests(unittest.TestCase):
    def test_message_width_stays_compatible(self):
        room = EscapeRoom()
        obs, _ = room.reset(1)
        self.assertEqual(room.message_count, 3)
        self.assertEqual(len(obs['A']), 21)

    def test_either_agent_can_unlock(self):
        for holder, activator, plate, switch in (
            (0, 1, (2, 2), (5, 2)),
            (1, 0, (2, 2), (5, 2)),
        ):
            env = RoleRoom(config=fixed_config())
            env.reset(1, clue=0, holder=0, swap_starts=False, stage='full')
            env.pos = [plate, switch] if holder == 0 else [switch, plate]
            actions = {'A': (4, 0), 'B': (4, 0)}
            actions[IDS[activator]] = (5, 0)
            _, reward, *_ = env.step(actions)
            self.assertTrue(env.unlocked)
            self.assertEqual(env.plate_holder, holder)
            self.assertEqual(env.switch_activator, activator)
            self.assertAlmostEqual(reward['A'], 0.99)

    def test_clue_stays_private(self):
        env = RoleRoom(config=fixed_config())
        for holder in (0, 1):
            env.reset(3, clue=0, holder=holder, swap_starts=False, stage='full')
            first = env.observations()
            env.clue = 1
            second = env.observations()
            reader, partner = ('A', 'B') if holder == 0 else ('B', 'A')
            np.testing.assert_equal(first[partner], second[partner])
            self.assertFalse(np.array_equal(first[reader], second[reader]))
            self.assertTrue(np.array_equal(first[reader][:BOARD], second[reader][:BOARD]))
            self.assertEqual(len(first['A']), OBS_SIZE)

    def test_one_step_delivery(self):
        env = RoleRoom(config=fixed_config())
        before, _ = env.reset(1, clue=0, holder=0, swap_starts=False, stage='full')
        self.assertEqual(list(before['A'][-MESSAGE_COUNT:]), [1, 0, 0, 0, 0])
        after, *_ = env.step({'A': (4, 4), 'B': (4, 1)})
        self.assertEqual(env.messages, [4, 1])
        self.assertEqual(list(after['B'][-MESSAGE_COUNT:]), [0, 0, 0, 0, 1])
        self.assertEqual(list(after['A'][-MESSAGE_COUNT:]), [0, 1, 0, 0, 0])

    def test_silent_channel_suppresses_both_symbols(self):
        env = RoleRoom(communication=False, config=fixed_config())
        env.reset(1, clue=1, holder=1, swap_starts=True, stage='full')
        obs, *_ = env.step({'A': (4, 4), 'B': (4, 3)})
        self.assertEqual(env.messages, [0, 0])
        self.assertEqual(list(obs['A'][-MESSAGE_COUNT:]), [1, 0, 0, 0, 0])
        self.assertEqual(list(obs['B'][-MESSAGE_COUNT:]), [1, 0, 0, 0, 0])
        self.assertEqual(RolePolicy().message.out_features, MESSAGE_COUNT)

    def test_curriculum_placement_and_eval_contract(self):
        self.assertEqual(stage_for(0), 'exit')
        self.assertEqual(stage_for(0.19), 'exit')
        self.assertEqual(stage_for(0.20), 'near')
        self.assertEqual(stage_for(0.49), 'near')
        self.assertEqual(stage_for(0.50), 'full')
        config = fixed_config()
        _, _, place = stream_rngs(11)
        exit_starts = sample_exit_starts(config, place)
        self.assertEqual(len(set(exit_starts)), 2)
        self.assertNotIn(tuple(config['exit']), exit_starts)
        near = sample_near_starts(config, place)
        self.assertEqual(len(set(near)), 2)
        self.assertTrue(set(near).isdisjoint(mechanisms(config)))
        env = RoleRoom(config=config)
        env.reset(11, stage='exit')
        self.assertTrue(env.unlocked)
        self.assertNotIn(tuple(env.pos[0]), {tuple(config['exit'])})
        env.reset(11, stage='near')
        self.assertFalse(env.unlocked)
        self.assertTrue(set(map(tuple, env.pos)).isdisjoint(mechanisms(config)))
        evaluated = reset_for_evaluation(config, clue=1, holder=0, swap=0, seed=4)
        self.assertEqual(evaluated.stage, 'full')
        self.assertFalse(evaluated.unlocked)
        self.assertEqual(evaluated.pos[0], tuple(config['starts'][0]))
        self.assertEqual(evaluated.pos[1], tuple(config['starts'][1]))
        swapped = reset_for_evaluation(config, clue=1, holder=1, swap=1, seed=4)
        self.assertEqual(swapped.pos[0], tuple(config['starts'][1]))
        self.assertEqual(swapped.clue_holder, 1)
        self.assertEqual(evaluated.pos, reset_for_evaluation(config, clue=0, holder=0, swap=0, seed=4).pos)
        base = EscapeRoom(config=config)
        base.reset(1, clue=0)
        role = RoleRoom(config=config)
        role.reset(1, clue=0, holder=0, swap_starts=False, stage='full')
        base.pos = [(2, 2), (5, 2)]
        role.pos = [(2, 2), (5, 2)]
        _, base_reward, *_ = base.step({'A': (4, 0), 'B': (5, 0)})
        _, role_reward, *_ = role.step({'A': (4, 0), 'B': (5, 0)})
        self.assertAlmostEqual(base_reward['A'], role_reward['A'])
        self.assertEqual(shaped_reward('full', False, False, True, role_reward['A'])[0], role_reward['A'])
        bonus, paid = shaped_reward('exit', False, False, True, role_reward['A'])
        self.assertAlmostEqual(bonus, role_reward['A'] + 1)
        self.assertTrue(paid)
        self.assertEqual(shaped_reward('exit', True, False, True, role_reward['A'])[0], role_reward['A'])
        cases = evaluation_cases(range(21000, 21002))
        self.assertEqual(len(cases), 16)
        self.assertEqual(cases, evaluation_cases(range(21000, 21002)))

    def test_plate_switch_exchange_keeps_the_rest(self):
        config = fixed_config()
        other = exchange_plate_and_switch(config, 0)
        self.assertEqual(other['plate'], config['switches'][0])
        self.assertEqual(other['switches'][0], config['plate'])
        self.assertEqual(other['switches'][1], config['switches'][1])
        self.assertEqual(other['exit'], config['exit'])
        self.assertEqual(other.get('walls', []), config.get('walls', []))
        self.assertEqual(other['starts'], config['starts'])


class PolicyTests(unittest.TestCase):
    def test_separate_memory_resets(self):
        torch.manual_seed(2)
        model = RolePolicy()
        board = torch.zeros(2, 7, 7, 9)
        board[0, 5, 3, 3] = 1
        board[1, 5, 4, 2] = 1
        extras = torch.zeros(2, EXTRAS)
        hidden = torch.zeros(2, HIDDEN)
        encoded_hidden = model.gru(model.encode(board, extras), hidden)
        self.assertFalse(torch.allclose(encoded_hidden[0], encoded_hidden[1]))
        reset = apply_done(encoded_hidden, torch.tensor([1.0, 0.0]))
        self.assertTrue(torch.equal(reset[0], torch.zeros(HIDDEN)))
        self.assertTrue(torch.equal(reset[1], encoded_hidden[1]))
        both = apply_done(encoded_hidden.reshape(1, 2, HIDDEN), torch.tensor([1.0]))
        self.assertTrue(torch.equal(both, torch.zeros_like(both)))

    def test_swapped_inputs_swap_distributions(self):
        torch.manual_seed(3)
        model = RolePolicy()
        board = torch.rand(2, 7, 7, 9)
        extras = torch.rand(2, EXTRAS)
        hidden = torch.randn(2, HIDDEN)
        encoded = model.encode(board, extras)
        updated = model.gru(encoded, hidden)
        first = model.heads(updated)
        flipped = model.heads(model.gru(model.encode(board.flip(0), extras.flip(0)), hidden.flip(0)))
        for original, swapped in zip(first, flipped):
            torch.testing.assert_close(original.flip(0), swapped)

    def test_gru_gradient_through_time_and_resets(self):
        torch.manual_seed(4)
        model = RolePolicy()
        boards = torch.rand(6, 3, 7, 7, 9)
        extras = torch.rand(6, 3, EXTRAS)
        actions = torch.randint(0, 5, (6, 3, 2))
        resets = torch.zeros(6, 3)
        resets[3, 1] = 1
        logp, entropy, value = sequence_forward(model, boards, extras, actions, resets, torch.zeros(3, HIDDEN))
        loss = -(logp * torch.randn_like(logp)).mean() + value.square().mean() - 0.01 * entropy.mean()
        loss.backward()
        self.assertGreater(model.gru.weight_hh.grad.abs().sum().item(), 0)
        self.assertGreater(model.conv[0].weight.grad.abs().sum().item(), 0)

    def test_batched_cases_match_solo_replay(self):
        torch.manual_seed(5)
        model = RolePolicy().eval()
        config = fixed_config()
        cases = [
            dict(config=config, layout_seed=4, clue=0, holder=0, swap=0, stream=17),
            dict(config=config, layout_seed=4, clue=1, holder=1, swap=1, stream=17),
        ]
        batched = run_cases(model, cases, communication=True)
        for case, episode in zip(cases, batched):
            env = reset_for_evaluation(config, case['clue'], case['holder'], case['swap'], seed=case['layout_seed'])
            actions = play(model, env, case_generator(case['layout_seed'], case['clue'], case['holder'], case['swap']))
            self.assertEqual(env.outcome, episode['outcome'])
            self.assertEqual(env.steps, episode['steps'])
            self.assertEqual(env.plate_holder, episode['plate_holder'])
            self.assertEqual(len(actions), episode['steps'])

    def test_short_update_changes_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            from role_ppo import train
            train(folder, seed=7, steps=32, communication=True, num_envs=2, rollout=8,
                  layouts=[fixed_config()], periodic_eval=False, final_eval=False)
            self.assertTrue((Path(folder) / 'checkpoint.pt').exists())
            self.assertTrue((Path(folder) / 'config.json').exists())


if __name__ == '__main__':
    unittest.main()
