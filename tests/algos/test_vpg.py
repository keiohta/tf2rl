import unittest

from tf2rl.algos.vpg import VPG
from tests.algos.common import CommonOnPolActorCriticContinuousAlgos, CommonOnPolActorCriticDiscreteAlgos


class TestContinuousVPG(CommonOnPolActorCriticContinuousAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = VPG(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            is_discrete=False,
            gpu=-1)


class TestDiscreteVPG(CommonOnPolActorCriticDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = VPG(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            is_discrete=True,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
