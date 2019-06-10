import unittest
import numpy as np
import tensorflow as tf

from tf2rl.algos.models import GaussianActor, CategoricalActor
from tests.algos.common import CommonAlgos


class TestGaussianActor(CommonAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.policy = GaussianActor(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            max_action=1.,
            units=[4,4])

    def test_call(self):
        states = np.random.rand(
            self.batch_size, self.continuous_env.observation_space.low.size)
        # Probabilistic sampling
        actions, log_probs = self.policy(states, test=False)
        self.assertEqual(
            actions.shape,
            (self.batch_size, self.continuous_env.action_space.low.size))
        self.assertEqual(
            log_probs.shape,
            (self.batch_size,))
        # Greedy sampling
        actions, log_probs = self.policy(states, test=True)
        self.assertEqual(
            actions.shape,
            (self.batch_size, self.continuous_env.action_space.low.size))
        self.assertEqual(
            log_probs.shape,
            (self.batch_size,))


class TestCategoricalActor(CommonAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.policy = CategoricalActor(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            units=[4,4])

    def test_call(self):
        # Single state
        states = np.random.rand(
            self.batch_size, self.discrete_env.observation_space.low.size)
        # Probabilistic sampling
        actions, log_probs = self.policy(states, test=False)
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(log_probs.shape, (self.batch_size,))
        # Greedy sampling
        actions, log_probs = self.policy(states, test=True)
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(log_probs.shape, (self.batch_size,))


if __name__ == '__main__':
    unittest.main()

