import unittest
from plate_signal import PlateRoom, load_tables, open_plate_room, plate_actions
from relay_room import RelayRoom


class PlateSignalTests(unittest.TestCase):
    def test_clue_reveal_message_and_escape(self):
        tables = load_tables()
        room = PlateRoom()
        for clue in (0, 1):
            obs, _ = room.reset(42, clue)
            events = {}
            while not room.done:
                actions = plate_actions(tables, obs, ())
                self.assertFalse(obs['A'][16:18].any())
                if not room.has_clue:
                    self.assertFalse(obs['B'][16:18].any())
                    self.assertEqual(actions['B'][1], 0)
                if not obs['A'][-2:].any() and not room.unlocked:
                    self.assertEqual(actions['A'][0], 4)
                obs, _, _, _, _ = room.step(actions)
                if room.has_clue:
                    events.setdefault('reveal', room.steps)
                    self.assertEqual(obs['B'][16 + clue], 1)
                if room.messages[1]:
                    events.setdefault('send', room.steps)
                if room.unlocked:
                    events.setdefault('unlock', room.steps)
            self.assertEqual(room.outcome, 'success')
            self.assertLess(events['reveal'], events['send'])
            self.assertLess(events['send'], events['unlock'])

    def _play(self, room, tables):
        obs = room.observations()
        events = {}
        while not room.done:
            self.assertFalse(obs['A'][16:18].any())
            actions = plate_actions(tables, obs, room.config.get('walls', ()))
            if not room.has_clue:
                self.assertFalse(obs['B'][16:18].any())
                self.assertEqual(actions['B'][1], 0)
            obs, _, _, _, _ = room.step(actions)
            if room.has_clue:
                events.setdefault('reveal', room.steps)
            if room.messages[1]:
                events.setdefault('send', room.steps)
            if room.unlocked:
                events.setdefault('unlock', room.steps)
        self.assertEqual(room.outcome, 'success')
        self.assertLess(events['reveal'], events['send'])
        self.assertLess(events['send'], events['unlock'])

    def test_random_seeds_escape_and_clear_the_relay(self):
        tables = load_tables()
        for seed in (0, 19, 42, 56, 132, 136, 143, 163, 999, 2**20, 2**31 - 1):
            room, _, used = open_plate_room(seed, tables=tables)
            self.assertEqual(used, seed)
            self.assertEqual(room.layout_seed, seed)
            self._play(room, tables)
            relay = RelayRoom(used)
            while not relay.done:
                relay.step(relay.reference_actions())
            self.assertEqual(relay.outcome, 'success')

    def test_unsolvable_clue_moves_to_a_later_seed(self):
        tables = load_tables()
        room, _, used = open_plate_room(56, clue=1, tables=tables)
        self.assertGreater(used, 56)
        self.assertEqual(room.clue, 1)
        self._play(room, tables)
        relay = RelayRoom(used)
        while not relay.done:
            relay.step(relay.reference_actions())
        self.assertEqual(relay.outcome, 'success')
