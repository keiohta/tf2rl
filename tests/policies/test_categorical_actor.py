import unittest
import numpy as np
import tensorflow as tf

from tf2rl.policies.categorical_actor import CategoricalActor
from tests.policies.common import CommonModel


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
        self._test_call(
            states, (self.batch_size,), (self.batch_size,))

    def test_compute_log_probs(self):
        # Single input
        state = np.random.rand(
            1, self.discrete_env.observation_space.low.size)
        action = np.random.rand(
            1, self.discrete_env.action_space.n)
        self._test_compute_log_probs(
            state, action, (1,))
        # Multiple inputs
        states = np.random.rand(
            self.batch_size, self.discrete_env.observation_space.low.size)
        actions = np.random.rand(
            self.batch_size, self.discrete_env.action_space.n)
        self._test_compute_log_probs(
            states, actions, (self.batch_size,))


if __name__ == '__main__':
    unittest.main()
