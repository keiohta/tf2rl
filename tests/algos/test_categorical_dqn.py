import unittest

from tf2rl.algos.categorical_dqn import CategoricalDQN
from tests.algos.common import CommonOffPolDiscreteAlgos


class TestCategoricalDQN(CommonOffPolDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = CategoricalDQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            epsilon=1.,
            gpu=-1)


class TestCategoricalDuelingDQN(CommonOffPolDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = CategoricalDQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            enable_dueling_dqn=True,
            epsilon=1.,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
