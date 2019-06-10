import unittest
import numpy as np
import tensorflow as tf

from tf2rl.algos.models import GaussianActor, CategoricalActor
from tests.algos.common import CommonAlgos


class CommonModel(CommonAlgos):
    def _test_call(self, inputs, expected_action_shapes, expected_log_prob_shapes):
        # Probabilistic sampling
        actions, log_probs = self.policy(inputs, test=False)
        self.assertEqual(actions.shape, expected_action_shapes)
        self.assertEqual(log_probs.shape, expected_log_prob_shapes)
        # Greedy sampling
        actions, log_probs = self.policy(inputs, test=True)
        self.assertEqual(actions.shape, expected_action_shapes)
        self.assertEqual(log_probs.shape, expected_log_prob_shapes)


class TestGaussianActor(CommonModel):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.policy = GaussianActor(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            max_action=1.,
            units=[4,4])

    def test_call(self):
        # Single input
        state = np.random.rand(
            1, self.continuous_env.observation_space.low.size)
        self._test_call(
            state,
            (1, self.continuous_env.action_space.low.size),
            (1,))
        # Multiple inputs
        states = np.random.rand(
            self.batch_size, self.continuous_env.observation_space.low.size)
        self._test_call(
            states,
            (self.batch_size, self.continuous_env.action_space.low.size),
            (self.batch_size,))


class TestCategoricalActor(CommonModel):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.policy = CategoricalActor(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            units=[4,4])

    def test_call(self):
        # Single input
        state = np.random.rand(
            1, self.discrete_env.observation_space.low.size)
        self._test_call(state, (1,), (1,))
        # Multiple inputs
        states = np.random.rand(
            self.batch_size, self.discrete_env.observation_space.low.size)
        self._test_call(states, (self.batch_size,), (self.batch_size,))


if __name__ == '__main__':
    unittest.main()

