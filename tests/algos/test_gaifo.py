import unittest

from tf2rl.algos.gaifo import GAIfO
from tests.algos.common import CommonIRLAlgos


class TestGAIfO(CommonIRLAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.irl_discrete = GAIfO(
            state_shape=cls.discrete_env.observation_space.shape,
            gpu=-1)
        cls.irl_continuous = GAIfO(
            state_shape=cls.continuous_env.observation_space.shape,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
