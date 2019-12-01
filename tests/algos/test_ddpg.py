import unittest

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
            sigma=0.5,  # Make noise bigger to easier to test
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
