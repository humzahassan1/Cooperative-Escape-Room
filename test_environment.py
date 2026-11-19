import unittest
import numpy as np
from environment import EscapeRoom, scripted

class RoomTests(unittest.TestCase):
    def setUp(self):self.e=EscapeRoom(); self.e.reset(7,clue=0)
    def step(self,a=4,b=4,ma=0,mb=0):return self.e.step({'A':(a,ma),'B':(b,mb)})
    def test_seed_and_privacy(self):
        a,_=self.e.reset(12); b,_=self.e.reset(12)
        np.testing.assert_equal(a['A'],b['A'])
        a,_=self.e.reset(12,clue=0); b,_=self.e.reset(12,clue=1)
        np.testing.assert_equal(a['B'],b['B']); self.assertFalse(np.array_equal(a['A'],b['A']))
    def test_messages(self):
        o,*_=self.step(ma=2); self.assertEqual(o['B'][-1],1)
        self.e.communication=False; o,*_=self.step(ma=2); self.assertEqual(o['B'][-3],1)
    def test_wall_exit(self):
        self.e.pos=[(1,1),(6,3)]; self.step(a=2,b=3)
        self.assertEqual(self.e.pos,[(1,1),(6,3)])
    def test_collisions(self):
        for positions,actions in [([(2,3),(4,3)],(3,2)), ([(2,3),(3,3)],(3,2)), ([(2,3),(3,3)],(3,4))]:
            self.e.reset(); self.e.pos=positions.copy(); self.step(*actions); self.assertEqual(self.e.pos,positions)
    def test_plate_and_unlock(self):
        self.e.pos=[(2,3),(5,2)]; self.step(b=5); self.assertFalse(self.e.unlocked)
        _,r,*_=self.step(a=0,b=5); self.assertTrue(self.e.unlocked); self.assertAlmostEqual(r['A'],.99)
        _,r,*_=self.step(a=1,b=5); self.assertTrue(self.e.unlocked); self.assertAlmostEqual(r['A'],-.01)
    def test_wrong_switch(self):
        self.e.pos=[(2,2),(5,4)]; _,r,t,tr,_=self.step(b=5)
        self.assertTrue(t); self.assertFalse(tr); self.assertAlmostEqual(r['A'],-1.01)
    def test_exit_and_done(self):
        self.e.unlocked=True; self.e.pos=[(6,3),(5,3)]; self.step(a=3)
        self.assertEqual(self.e.escaped,[True,False]); self.step(b=3)
        _,r,t,_,_=self.step(b=3); self.assertTrue(t); self.assertAlmostEqual(r['A'],4.99)
        with self.assertRaises(RuntimeError):self.step()
    def test_timeout(self):
        for _ in range(100):o,r,t,tr,info=self.step()
        self.assertFalse(t); self.assertTrue(tr)
    def test_scripted(self):
        for seed in range(200):
            obs,_=self.e.reset(seed,clue=seed%2)
            while not self.e.done:obs,*_=self.e.step(scripted(obs))
            self.assertEqual(self.e.outcome,'success')

if __name__=='__main__':unittest.main()
