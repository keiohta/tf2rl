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
        cls.batch_size = 32
        cls.dummy_continuous_obses = np.zeros(
            shape=(cls.batch_size,)+cls.continuous_env.observation_space.shape,
            dtype=np.float32)
        cls.dummy_continuous_obs = np.copy(
            cls.dummy_continuous_obses[0])

    def test__init__(self):
        # VPG(state_shape=self.discrete_env.observation_space.shape,
        #     action_dim=self.discrete_env.action_space.n,
        #     is_discrete=True, gpu=-1)
        VPG(state_shape=self.continuous_env.observation_space.shape,
            action_dim=self.continuous_env.action_space.low.size,
            is_discrete=False, gpu=-1)

    def test_get_action(self):
        # discrete_vpg = VPG(
        #     state_shape=self.discrete_env.observation_space.shape,
        #     action_dim=self.discrete_env.action_space.n,
        #     is_discrete=True, gpu=-1)
        # discrete_obs = self.discrete_env.reset()
        # discrete_vpg.get_action(discrete_obs, test=False)
        # discrete_vpg.get_action(discrete_obs, test=True)

        continuous_vpg = VPG(
            state_shape=self.continuous_env.observation_space.shape,
            action_dim=self.continuous_env.action_space.low.size,
            is_discrete=False, gpu=-1)
        # continuous_obs = self.continuous_env.reset()
        action, log_pi = continuous_vpg.get_action(
            self.dummy_continuous_obs, test=False)
        self.assertEqual(action.ndim, 1)
        actions, log_pis = continuous_vpg.get_action(
            self.dummy_continuous_obses, test=False)
        self.assertEqual(actions.shape[1],
                         self.continuous_env.action_space.low.size)
        self.assertEqual(actions.shape[0],
                         self.batch_size)

    def test_train(self):
        rewards = np.zeros(shape=(self.batch_size,), dtype=np.float32)
        dones = np.zeros(shape=(self.batch_size,), dtype=np.float32)
        log_pis = np.zeros(shape=(self.batch_size,), dtype=np.float32)

        print("Discrete test is not implemented yet")
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
        continuous_acts = np.zeros(
            shape=(self.batch_size,)+self.continuous_env.action_space.shape,
            dtype=np.float32)
        continuous_vpg.train_actor(
            self.dummy_continuous_obses, continuous_acts,
            self.dummy_continuous_obses, rewards, dones, log_pis)
        continuous_vpg.train_critic(
            self.dummy_continuous_obses, continuous_acts,
            self.dummy_continuous_obses, rewards, dones)


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.enable_eager_execution(config=config)
    unittest.main()
