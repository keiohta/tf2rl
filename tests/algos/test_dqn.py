import unittest
import numpy as np
import tensorflow as tf

from tf2rl.algos.dqn import DQN
from tests.algos.common import CommonDiscreteOutputAlgos


class TestDQN(CommonDiscreteOutputAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            gpu=-1)


class TestDuelingDoubleDQN(CommonDiscreteOutputAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            enable_double_dqn=True,
            enable_dueling_dqn=True,
            gpu=-1)


class TestNoisyDQN(CommonDiscreteOutputAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            enable_noisy_dqn=True,
            gpu=-1)


class TestCategoricalDQN(CommonDiscreteOutputAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            enable_categorical_dqn=True,
            enable_dueling_dqn=True,
            gpu=-1)


class TestCategoricalDuelingDQN(CommonDiscreteOutputAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.agent = DQN(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            batch_size=cls.batch_size,
            enable_categorical_dqn=True,
            gpu=-1)


if __name__ == '__main__':
    unittest.main()
