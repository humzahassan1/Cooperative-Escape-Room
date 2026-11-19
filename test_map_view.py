import unittest
import numpy as np
from map_ppo import MapRoom, SIZE
from environment import EscapeRoom


class FullMapTests(unittest.TestCase):
    def test_public_geometry_and_private_clue(self):
        config=dict(EscapeRoom().config,interchangeable=True,walls=[[4,3]])
        room=MapRoom(config=config)
        for holder in (0,1):
            room.clue_holder=holder
            room.clue=0
            a=room.observations()
            room.clue=1
            b=room.observations()
            reader,partner=('A','B') if holder==0 else ('B','A')
            np.testing.assert_equal(a[partner],b[partner])
            self.assertFalse(np.array_equal(a[reader],b[reader]))
            for i,name in enumerate(('A','B')):
                self.assertEqual(len(a[name]),SIZE)
                board=a[name][:-9].reshape(7,7,9)
                self.assertEqual(board[0,3,4],1)
                objects=[config['plate'],*config['switches'],config['exit'],room.pos[i],room.pos[1-i]]
                for layer,(x,y) in enumerate(objects,1):
                    self.assertEqual(board[layer,y,x],1)
        alternate=MapRoom(config=dict(config,walls=[[3,4]]))
        altered=alternate.observations()['A'][:63]
        self.assertFalse(np.array_equal(altered,a['A'][:63]))
