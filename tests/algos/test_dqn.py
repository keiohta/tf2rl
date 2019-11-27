import unittest

import numpy as np

from tf2rl.algos.dqn import DQN
from tests.algos.common import CommonOffPolDiscreteAlgos


class TestDQN(CommonOffPolDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            epsilon=1.,
            gpu=-1)


class TestDuelingDoubleDQN(CommonOffPolDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            enable_double_dqn=True,
            enable_dueling_dqn=True,
            epsilon=1.,
            gpu=-1)


class TestNoisyDQN(CommonOffPolDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            enable_noisy_dqn=True,
            epsilon=1.,
            gpu=-1)

    def get_actions(self, states, test):
        if test:
            return self.agent.get_action(states, test=test)
        else:
            return np.array([self.agent.get_action(state) for state in states])


class TestCategoricalDQN(CommonOffPolDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            enable_categorical_dqn=True,
            enable_dueling_dqn=True,
            epsilon=1.,
            gpu=-1)


class TestCategoricalDuelingDQN(CommonOffPolDiscreteAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            enable_categorical_dqn=True,
            epsilon=1.,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
