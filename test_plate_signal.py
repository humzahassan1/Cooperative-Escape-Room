import unittest
from plate_signal import PlateRoom, load_tables, plate_actions


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
