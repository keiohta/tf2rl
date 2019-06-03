import unittest
import gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.dqn import DQN
from tests.algos.common import CommonAlgos


# class TestDQN(unittest.TestCase):
class TestDQN(CommonAlgos):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("CartPole-v0")
        cls.batch_size = 32

    def test__init__(self):
        DQN(state_shape=self.env.observation_space.shape,
            action_dim=self.env.action_space.n,
            batch_size=self.batch_size,
            gpu=-1)

    def test_get_action(self):
        agent = DQN(
            state_shape=self.env.observation_space.shape,
            action_dim=self.env.action_space.n,
            batch_size=self.batch_size,
            gpu=-1)
        state = self.env.reset()
        agent.get_action(state, test=False)
        agent.get_action(state, test=True)

    def test_train(self):
        agent = DQN(
            state_shape=self.env.observation_space.shape,
            action_dim=self.env.action_space.n,
            batch_size=self.batch_size,
            gpu=-1)

        rewards = np.zeros(shape=(self.batch_size,1), dtype=np.float32)
        dones = np.zeros(shape=(self.batch_size,1), dtype=np.float32)
        obses = np.zeros(
            shape=(self.batch_size,)+self.env.observation_space.shape,
            dtype=np.float32)
        acts = np.zeros(shape=(self.batch_size,1), dtype=np.float32)
        agent.train(
            obses, acts, obses, rewards, dones)


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.enable_eager_execution(config=config)
    unittest.main()
