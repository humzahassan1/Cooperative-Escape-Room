import unittest
from relay_room import RelayRoom, PASS


class RelayTests(unittest.TestCase):
    def test_no_remote_pickup_pass_or_crossing(self):
        env=RelayRoom(7)
        env.pos=[(3,3),(6,3)]
        env.step({'A':(3,0),'B':(5,0)})
        self.assertEqual(env.pos[0],(3,3))
        self.assertIsNone(env.carrier)
        env.carrier=0
        env.step({'A':(PASS,0),'B':(4,0)})
        self.assertEqual(env.carrier,0)
        env.pos[1]=(5,3)
        env.step({'A':(PASS,0),'B':(PASS,0)})
        self.assertEqual(env.carrier,1)
        self.assertFalse(env.unlocked)

    def test_reference_completes_handoff_and_escape(self):
        owners=set()
        for seed in range(100):
            room=RelayRoom(seed)
            transfers=0
            while not room.done:
                old=room.carrier
                obs,_,_,_,_=room.step(room.reference_actions())
                self.assertEqual(obs['A'].shape,(450,))
                if old is not None and room.carrier is not None and old!=room.carrier:
                    owners.add(old); transfers+=1
            self.assertEqual(transfers,1)
            self.assertTrue(room.unlocked)
            self.assertEqual(room.outcome,'success')
        self.assertEqual(owners,{0,1})
