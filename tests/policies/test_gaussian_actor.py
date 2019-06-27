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
            units=[4, 4])
        cls.const_std = 0.1
        cls.policy_fixed_sigma = GaussianActor(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            max_action=1.,
            units=[4, 4],
            fix_std=True,
            const_std=cls.const_std)

    def test_call(self):
        """Not fix sigma"""
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

        """Fix sigma"""
        state = np.random.rand(
            1, self.continuous_env.observation_space.low.size)
        self._test_call(
            state,
            (1, self.continuous_env.action_space.low.size),
            (1,),
            policy=self.policy_fixed_sigma)
        # Multiple inputs
        states = np.random.rand(
            self.batch_size, self.continuous_env.observation_space.low.size)
        self._test_call(
            states,
            (self.batch_size, self.continuous_env.action_space.low.size),
            (self.batch_size,),
            policy=self.policy_fixed_sigma)

    def test_compute_log_probs(self):
        """Not fix sigma"""
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

        """Fix sigma"""
        # Single input
        state = np.random.rand(
            1, self.continuous_env.observation_space.low.size).astype(np.float32)
        action = np.random.rand(
            1, self.continuous_env.action_space.low.size).astype(np.float32)
        self._test_compute_log_probs(
            state, action, (1,),
            policy=self.policy_fixed_sigma)
        # Multiple inputs
        states = np.random.rand(
            self.batch_size,
            self.continuous_env.observation_space.low.size).astype(np.float32)
        actions = np.random.rand(
            self.batch_size,
            self.continuous_env.action_space.low.size).astype(np.float32)
        self._test_compute_log_probs(
            states, actions, (self.batch_size,),
            policy=self.policy_fixed_sigma)


if __name__ == '__main__':
    unittest.main()
