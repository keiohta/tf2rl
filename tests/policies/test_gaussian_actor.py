import unittest
import numpy as np
import tensorflow as tf

from tf2rl.policies.gaussian_actor import GaussianActor
from tests.policies.common import CommonModel


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

    def test_compute_log_probs(self):
        # Single input
        state = np.random.rand(
            1, self.continuous_env.observation_space.low.size).astype(np.float32)
        action = np.random.rand(
            1, self.continuous_env.action_space.low.size).astype(np.float32)
        self._test_compute_log_probs(
            state, action, (1,))
        # Multiple inputs
        states = np.random.rand(
            self.batch_size,
            self.continuous_env.observation_space.low.size).astype(np.float32)
        actions = np.random.rand(
            self.batch_size,
            self.continuous_env.action_space.low.size).astype(np.float32)
        self._test_compute_log_probs(
            states, actions, (self.batch_size,))


if __name__ == '__main__':
    unittest.main()
