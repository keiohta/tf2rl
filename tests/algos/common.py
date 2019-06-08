import unittest
import numpy as np
import gym


class CommonAlgos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TODO: Remove dependencies to gym
        cls.discrete_env = gym.make("CartPole-v0")
        cls.continuous_env = gym.make("Pendulum-v0")
        cls.batch_size = 32
        cls.agent = None


class CommonContinuousOutputAlgos(CommonAlgos):
    def test_get_action(self):
        state = self.continuous_env.reset()
        if self.agent is None:
            return
        self.agent.get_action(state, test=False)
        self.agent.get_action(state, test=True)

    def test_train(self):
        if self.agent is None:
            return

        rewards = np.zeros(shape=(self.batch_size,1), dtype=np.float32)
        dones = np.zeros(shape=(self.batch_size,1), dtype=np.float32)
        obses = np.zeros(
            shape=(self.batch_size,)+self.continuous_env.observation_space.shape,
            dtype=np.float32)
        acts = np.zeros(
            shape=(self.batch_size,self.continuous_env.action_space.low.size,),
            dtype=np.float32)
        self.agent.train(
            obses, acts, obses, rewards, dones)


class CommonDiscreteOutputAlgos(CommonAlgos):
    def test_get_action(self):
        if self.agent is None:
            return
        state = self.discrete_env.reset()
        self.agent.get_action(state, test=False)
        self.agent.get_action(state, test=True)

    def test_train(self):
        if self.agent is None:
            return

        rewards = np.zeros(
            shape=(self.batch_size,1),
            dtype=np.float32)
        dones = np.zeros(
            shape=(self.batch_size,1),
            dtype=np.float32)
        obses = np.zeros(
            shape=(self.batch_size,)+self.discrete_env.observation_space.shape,
            dtype=np.float32)
        acts = np.zeros(
            shape=(self.batch_size,1),
            dtype=np.float32)
        self.agent.train(
            obses, acts, obses, rewards, dones)


if __name__ == '__main__':
    unittest.main()
