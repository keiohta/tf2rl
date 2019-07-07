import unittest
import gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.vail import VAIL
from tests.algos.common import CommonIRLAlgos


class TestVAIL(CommonIRLAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.irl_discrete = VAIL(
            state_shape=cls.discrete_env.observation_space.shape,
            action_dim=cls.discrete_env.action_space.n,
            gpu=-1)
        cls.irl_continuous = VAIL(
            state_shape=cls.continuous_env.observation_space.shape,
            action_dim=cls.continuous_env.action_space.low.size,
            gpu=-1)

    def test__compute_kl_latent(self):
        means = np.zeros(
            shape=(self.batch_size, self.continuous_env.action_space.high.size))
        log_stds = np.zeros_like(means)
        results = self.irl_continuous._compute_kl_latent(
            means=means, log_stds=log_stds)
        np.testing.assert_array_equal(results, np.zeros_like(results))

        means = np.random.rand(
            self.batch_size, self.continuous_env.action_space.high.size)
        log_stds = np.random.rand(
            self.batch_size, self.continuous_env.action_space.high.size)
        results = self.irl_continuous._compute_kl_latent(
            means=means, log_stds=log_stds)
        np.testing.assert_equal(np.any(np.not_equal(
            results, np.zeros_like(results))), True)


if __name__ == '__main__':
    unittest.main()
