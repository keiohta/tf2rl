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


if __name__ == '__main__':
    unittest.main()
