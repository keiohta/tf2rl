import unittest

from tf2rl.algos.airl import AIRL
from tests.algos.common import CommonIRLAlgos


class TestAIRLState(CommonIRLAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.irl_discrete = AIRL(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            state_only=True,
            gpu=-1)
        cls.irl_continuous = AIRL(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            state_only=True,
            gpu=-1)


class TestAIRLStateAct(CommonIRLAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.irl_discrete = AIRL(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            state_only=False,
            gpu=-1)
        cls.irl_continuous = AIRL(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            state_only=False,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
