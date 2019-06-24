import unittest
import numpy as np
import tensorflow as tf

from tf2rl.algos.td3 import TD3
from tests.algos.common import CommonOffPolContinuousAlgos


class TestTD3(CommonOffPolContinuousAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = TD3(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            batch_size=cls.batch_size,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
