import unittest
import gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.gail import GAIL
from tests.algos.common import CommonIRLAlgos


class TestGAIL(CommonIRLAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.irl_discrete = GAIL(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            gpu=-1)
        cls.irl_continuous = GAIL(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            gpu=-1)

    # def test_train(self):
    #     gail = GAIL(
    #         state_shape=self.env.observation_space.shape,
    #         action_dim=self.env.action_space.n,
    #         gpu=-1)
    #     states = np.zeros(
    #         shape=(self.batch_size, self.env.observation_space.low.size),
    #         dtype=np.float32)
    #     actions = np.zeros(
    #         shape=(self.batch_size, self.env.action_space.n),
    #         dtype=np.float32)
    #     gail.train(states, actions, states, actions)

    # def test_inference(self):
    #     gail = GAIL(
    #         state_shape=self.env.observation_space.shape,
    #         action_dim=self.env.action_space.n,
    #         gpu=-1)
    #     state = np.zeros(
    #         shape=(self.env.observation_space.low.size,), dtype=np.float32)
    #     action = np.zeros(shape=(self.env.action_space.n,), dtype=np.float32)
    #     action[self.env.action_space.sample()] = 1.
    #     gail.inference(state, action)


if __name__ == '__main__':
    unittest.main()
