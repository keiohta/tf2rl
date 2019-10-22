import unittest

from tf2rl.algos.sac_discrete import SACDiscrete
from tests.algos.common import CommonOffPolDiscreteAlgos


class TestSAC(CommonOffPolDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = SACDiscrete(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
