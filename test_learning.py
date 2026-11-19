import io
from pathlib import Path
import unittest
import torch
from environment import EscapeRoom, OBS_VERSION
from learning import Policy, compute_gae, load

class LearningTests(unittest.TestCase):
    def test_terminal_timeout_and_reset_boundaries(self):
        rewards=torch.tensor([[[1.,1.]],[[100.,100.]]])
        values=torch.zeros_like(rewards)
        next_values=torch.tensor([[[0.,2.]],[[0.,0.]]])
        ends=torch.ones((2,1))
        advantage,returns=compute_gae(rewards,values,next_values,ends)
        self.assertAlmostEqual(returns[0,0,0].item(),1.)
        self.assertAlmostEqual(returns[0,0,1].item(),2.98,places=5)
    def test_update_and_checkpoint_roundtrip(self):
        torch.manual_seed(7); p=Policy(); opt=torch.optim.Adam(p.parameters(),lr=.0003)
        obs=torch.randn(32,21); before=p.actor.weight.detach().clone()
        with torch.no_grad():action,old,_,_=p.sample(obs)
        _,logprob,entropy,value=p.sample(obs,action)
        advantages=torch.linspace(-1,1,32); ratio=(logprob-old).exp()
        loss=torch.maximum(-advantages*ratio,-advantages*ratio.clamp(.8,1.2)).mean()+.5*(value-1).square().mean()-.01*entropy.mean()
        self.assertTrue(torch.isfinite(loss)); loss.backward(); opt.step()
        self.assertFalse(torch.equal(before,p.actor.weight))
        with io.BytesIO() as path:
            torch.save(dict(policies=[p.state_dict(),p.state_dict()],observation_version=OBS_VERSION,config={'room':EscapeRoom().config}),path)
            path.seek(0)
            restored,_=load(path)
            with torch.no_grad():
                torch.testing.assert_close(p.sample(obs,action)[1],restored[0].sample(obs,action)[1])
    def test_delivered_checkpoints(self):
        for folder in ['communication','no_communication']:
            path=Path(__file__).parent/'runs'/folder/'checkpoint.pt'
            if not path.exists():continue
            policies,data=load(path); self.assertEqual(data['steps'],501760)
            self.assertTrue(all(torch.isfinite(t).all() for p in policies for t in p.parameters()))

if __name__=='__main__':unittest.main()
