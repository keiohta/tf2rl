import unittest

import os
import numpy as np
import gym

from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.experiments.utils import save_path, restore_latest_n_traj
from tf2rl.algos.dqn import DQN


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("CartPole-v0")
        policy = DQN(
            state_shape=cls.env.observation_space.shape,
            action_dim=cls.env.action_space.n,
            memory_capacity=2**4)
        cls.replay_buffer = get_replay_buffer(
            policy, cls.env)
        cls.output_dir = os.path.join(
            os.path.dirname(__file__),
            "tests")
        if not os.path.isdir(cls.output_dir):
            os.makedirs(cls.output_dir)

    def test_save_path(self):
        n_store_episodes = 10
        obs = np.ones(shape=self.env.observation_space.shape, dtype=np.float32)
        for epi in range(n_store_episodes):
            for i in range(self.replay_buffer.get_buffer_size()):
                self.replay_buffer.add(
                    obs=obs*i, act=i, rew=0., next_obs=obs*(i+1), done=False)
            save_path(
                self.replay_buffer.sample(
                    self.replay_buffer.get_buffer_size()),
                os.path.join(self.output_dir,
                             "step_0_epi_{}_return_0.0.pkl").format(epi))
        data = restore_latest_n_traj(self.output_dir)
        self.assertEqual(data["obses"].shape[0],
                         self.replay_buffer.get_buffer_size() * n_store_episodes)


if __name__ == '__main__':
    unittest.main()
