import unittest
import numpy as np
import tensorflow as tf

from tf2rl.algos.ddpg import DDPG
from tests.algos.common import CommonOffPolContinuousAlgos


class TestDDPG(CommonOffPolContinuousAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DDPG(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            batch_size=cls.batch_size,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
