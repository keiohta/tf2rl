import unittest
import gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.vpg import VPG


class VPGTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.discrete_env = gym.make("CartPole-v0")
        cls.continuous_env = gym.make("Pendulum-v0")

    def test__init__(self):
        VPG(state_shape=self.discrete_env.observation_space.shape,
            action_dim=self.discrete_env.action_space.n,
            is_discrete=True, gpu=-1)
        VPG(state_shape=self.continuous_env.observation_space.shape,
            action_dim=self.continuous_env.action_space.low.size,
            is_discrete=False, gpu=-1)

    def test_get_action(self):
        discrete_vpg = VPG(
            state_shape=self.discrete_env.observation_space.shape,
            action_dim=self.discrete_env.action_space.n,
            is_discrete=True, gpu=-1)
        discrete_obs = self.discrete_env.reset()
        discrete_vpg.get_action(discrete_obs, test=False)
        discrete_vpg.get_action(discrete_obs, test=True)

        continuous_vpg = VPG(state_shape=self.continuous_env.observation_space.shape,
            action_dim=self.continuous_env.action_space.low.size,
            is_discrete=False, gpu=-1)
        continuous_obs = self.continuous_env.reset()
        continuous_vpg.get_action(continuous_obs, test=False)
        continuous_vpg.get_action(continuous_obs, test=True)

    def test_train(self):
        batch_size = 32
        rewards = np.zeros(shape=(batch_size,))
        dones = np.zeros(shape=(batch_size,))

        # discrete_vpg = VPG(
        #     state_shape=self.discrete_env.observation_space.shape,
        #     action_dim=self.discrete_env.action_space.n,
        #     is_discrete=True, gpu=-1)
        # discrete_obses = np.zeros(
        #     shape=(batch_size,)+self.discrete_env.observation_space.shape)
        # discrete_acts = np.zeros(batch_size)
        # discrete_vpg.train(
        #     discrete_obses, discrete_acts, discrete_obses, rewards, dones)

        continuous_vpg = VPG(
            state_shape=self.continuous_env.observation_space.shape,
            action_dim=self.continuous_env.action_space.low.size,
            is_discrete=False, gpu=-1)
        continuous_obses = np.zeros(
            shape=(batch_size,)+self.continuous_env.observation_space.shape)
        continuous_acts = np.zeros(
            shape=(batch_size,)+self.continuous_env.action_space.shape)
        continuous_vpg.train(
            continuous_obses, continuous_acts, continuous_obses, rewards, dones)


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.enable_eager_execution(config=config)
    unittest.main()
