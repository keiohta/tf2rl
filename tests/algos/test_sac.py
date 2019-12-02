import unittest

from tf2rl.algos.sac import SAC
from tests.algos.common import CommonOffPolContinuousAlgos


class TestSAC(CommonOffPolContinuousAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = SAC(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            batch_size=cls.batch_size,
            gpu=-1)


class TestSACAutoAlpha(CommonOffPolContinuousAlgos):
    # TODO: Skip duplicated tests called in TestSAC
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = SAC(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            batch_size=cls.batch_size,
            auto_alpha=True,
            gpu=-1)



if __name__ == '__main__':
    unittest.main()
