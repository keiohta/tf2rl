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
        if self.agent is None:
            return
        # Single input
        state = self.continuous_env.reset()
        action = self.agent.get_action(state, test=False)
        self.assertEqual(
            action.shape[0],
            self.continuous_env.action_space.low.size)
        action = self.agent.get_action(state, test=True)
        self.assertEqual(
            action.shape[0],
            self.continuous_env.action_space.low.size)

        # Multiple inputs
        states = np.zeros(shape=(self.batch_size, state.shape[0]))
        actions = self.agent.get_action(states, test=False)
        self.assertEqual(
            actions.shape[0],
            self.batch_size)
        self.assertEqual(
            actions.shape[1],
            self.continuous_env.action_space.low.size)
        actions = self.agent.get_action(states, test=True)
        self.assertEqual(
            actions.shape[0],
            self.batch_size)
        self.assertEqual(
            actions.shape[1],
            self.continuous_env.action_space.low.size)

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

    def test_compute_td_error(self):
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
        self.agent.compute_td_error(
            states=obses, actions=acts, next_states=obses,
            rewards=rewards, dones=dones)


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


class CommonIRLAlgos(CommonAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.irl_discrete = None
        cls.irl_continuous = None

    def test_inference_discrete(self):
        if self.irl_discrete is None:
            return
        state = np.zeros(
            shape=(self.discrete_env.observation_space.low.size,),
            dtype=np.float32)
        action = np.zeros(
            shape=(self.discrete_env.action_space.n,),
            dtype=np.float32)
        action[self.discrete_env.action_space.sample()] = 1.
        self.irl_discrete.inference(state, action)

    def test_inference_continuous(self):
        if self.irl_continuous is None:
            return
        state = np.zeros(
            shape=(self.continuous_env.observation_space.low.size,),
            dtype=np.float32)
        action = np.zeros(
            shape=(self.continuous_env.action_space.low.size,),
            dtype=np.float32)
        self.irl_continuous.inference(state, action)

    def test_train_discrete(self):
        if self.irl_discrete is None:
            return
        states = np.zeros(
            shape=(self.batch_size, self.discrete_env.observation_space.low.size),
            dtype=np.float32)
        actions = np.zeros(
            shape=(self.batch_size, self.discrete_env.action_space.n),
            dtype=np.float32)
        self.irl_discrete.train(states, actions, states, actions)

    def test_train_continuous(self):
        if self.irl_continuous is None:
            return
        states = np.zeros(
            shape=(self.batch_size, self.continuous_env.observation_space.low.size),
            dtype=np.float32)
        actions = np.zeros(
            shape=(self.batch_size, self.continuous_env.action_space.low.size),
            dtype=np.float32)
        self.irl_continuous.train(states, actions, states, actions)


if __name__ == '__main__':
    unittest.main()
