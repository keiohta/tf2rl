import unittest

from tf2rl.algos.airl import AIRL
from tests.algos.common import CommonIRLAlgos


class TestGAIL(CommonIRLAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.irl_discrete_state = AIRL(
            state_shape=cls.discrete_env.observation_space.shape,
            gpu=-1)
        cls.irl_continuous_state = AIRL(
            state_shape=cls.continuous_env.observation_space.shape,
            gpu=-1)
        cls.irl_discrete_state_act = AIRL(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            gpu=-1)
        cls.irl_continuous_state_act = AIRL(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
