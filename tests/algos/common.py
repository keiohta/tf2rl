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


class CommonOffPolAlgos(CommonAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env = None
        cls.action_dim = None
        cls.is_discrete = True

    def test_get_action(self):
        if self.agent is None:
            return
        # Single input
        state = self.env.reset().astype(np.float32)
        action_train = self.agent.get_action(state, test=False)
        action_test = self.agent.get_action(state, test=True)
        if self.is_discrete:
            self.assertTrue(isinstance(action_train, (np.int32, np.int64, int)))
            self.assertTrue(isinstance(action_test, (np.int32, np.int64, int)))
        else:
            self.assertEqual(action_train.shape[0], self.action_dim)
            self.assertEqual(action_test.shape[0], self.action_dim)

        # Multiple inputs
        states = np.zeros(
            shape=(self.batch_size, state.shape[0]), dtype=np.float32)
        actions_train = self.agent.get_action(states, test=False)
        actions_test = self.agent.get_action(states, test=True)

        if self.is_discrete:
            self.assertEqual(
                actions_train.shape, (self.batch_size,))
            self.assertEqual(
                actions_test.shape, (self.batch_size,))
        else:
            self.assertEqual(
                actions_train.shape, (self.batch_size, self.action_dim))
            self.assertEqual(
                actions_test.shape, (self.batch_size, self.action_dim))

    def test_get_action_greedy(self):
        if self.agent is None:
            return
        # Multiple inputs
        states = np.zeros(
            shape=(self.batch_size, self.env.reset().astype(np.float32).shape[0]), dtype=np.float32)
        actions_train = self.agent.get_action(states, test=False)
        actions_test = self.agent.get_action(states, test=True)

        # All actions should be same if `test=True`, and not same if `test=False`
        if self.is_discrete:
            self.assertEqual(np.prod(np.unique(actions_test).shape), 1)
            self.assertGreater(np.prod(np.unique(actions_train).shape), 1)
        else:
            self.assertEqual(np.prod(np.all(actions_test == actions_test[0, :], axis=0)), 1)
            self.assertEqual(np.prod(np.all(actions_train == actions_train[0, :], axis=0)), 0)

    def test_train(self):
        if self.agent is None:
            return
        rewards = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        dones = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        obses = np.zeros(
            shape=(self.batch_size,)+self.env.observation_space.shape,
            dtype=np.float32)
        acts = np.zeros(
            shape=(self.batch_size, self.action_dim,),
            dtype=np.float32)
        self.agent.train(
            obses, acts, obses, rewards, dones)

    def test_compute_td_error(self):
        if self.agent is None:
            return
        rewards = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        dones = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)
        obses = np.zeros(
            shape=(self.batch_size,)+self.env.observation_space.shape,
            dtype=np.float32)
        acts = np.zeros(
            shape=(self.batch_size, self.continuous_env.action_space.low.size,),
            dtype=np.float32)
        self.agent.compute_td_error(
            states=obses, actions=acts, next_states=obses,
            rewards=rewards, dones=dones)


class CommonOffPolContinuousAlgos(CommonOffPolAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env = cls.continuous_env
        cls.action_dim = cls.continuous_env.action_space.low.size
        cls.is_discrete = False


class CommonOffPolDiscreteAlgos(CommonOffPolAlgos):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env = cls.discrete_env
        cls.action_dim = 1
        cls.is_discrete = True


class CommonOnPolActorCritic(CommonAlgos):
    def test_get_action(self):
        if self.agent is None:
            return
        # Single input
        state = self.env.reset().astype(np.float32)
        action_train, logp_train = self.agent.get_action(state, test=False)
        action_test, logp_test = self.agent.get_action(state, test=True)
        if self.is_discrete:
            self.assertTrue(isinstance(action_train, (np.int32, np.int64, int)))
            self.assertTrue(isinstance(action_test, (np.int32, np.int64, int)))
        else:
            self.assertEqual(action_train.shape[0], self.action_dim)
            self.assertEqual(action_test.shape[0], self.action_dim)
        self.assertEqual(logp_train.shape[0], 1)
        self.assertEqual(logp_test.shape[0], 1)

        # Multiple inputs
        states = np.zeros(
            shape=(self.batch_size, state.shape[0]), dtype=np.float32)
        actions_train, logps_train = self.agent.get_action(states, test=False)
        actions_test, logps_test = self.agent.get_action(states, test=True)
        if self.is_discrete:
            self.assertEqual(
                actions_train.shape, (self.batch_size,))
            self.assertEqual(
                actions_test.shape, (self.batch_size,))
        else:
            self.assertEqual(
                actions_train.shape, (self.batch_size, self.action_dim))
            self.assertEqual(
                actions_test.shape, (self.batch_size, self.action_dim))
        self.assertEqual(logps_train.shape, (self.batch_size,))
        self.assertEqual(logps_test.shape, (self.batch_size,))

    def test_train(self):
        if self.agent is None:
            return
        state = self.env.reset().astype(np.float32)
        obses = np.zeros(
            shape=(self.batch_size,)+state.shape,
            dtype=np.float32)
        acts = np.zeros(
            shape=(self.batch_size, self.action_dim),
            dtype=np.int32 if self.is_discrete else np.float32)
        advs = np.ones(
            shape=(self.batch_size, 1),
            dtype=np.float32)
        logps = np.ones_like(advs)
        returns = np.zeros(
            shape=(self.batch_size, 1),
            dtype=np.float32)

        self.agent.train(obses, acts, advs, logps, returns)


class CommonOnPolActorCriticContinuousAlgos(CommonOnPolActorCritic):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env = cls.continuous_env
        cls.action_dim = cls.continuous_env.action_space.low.size
        cls.is_discrete = False


class CommonOnPolActorCriticDiscreteAlgos(CommonOnPolActorCritic):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env = cls.discrete_env
        cls.action_dim = 1
        cls.is_discrete = True


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
        self.irl_discrete.inference(state, action, state)

    def test_inference_continuous(self):
        if self.irl_continuous is None:
            return
        state = np.zeros(
            shape=(self.continuous_env.observation_space.low.size,),
            dtype=np.float32)
        action = np.zeros(
            shape=(self.continuous_env.action_space.low.size,),
            dtype=np.float32)
        self.irl_continuous.inference(state, action, state)

    def test_train_discrete(self):
        if self.irl_discrete is None:
            return
        states = np.zeros(
            shape=(self.batch_size, self.discrete_env.observation_space.low.size),
            dtype=np.float32)
        actions = np.zeros(
            shape=(self.batch_size, self.discrete_env.action_space.n),
            dtype=np.float32)
        self.irl_discrete.train(
            agent_states=states,
            agent_acts=actions,
            agent_next_states=states,
            expert_states=states,
            expert_acts=actions,
            expert_next_states=states)

    def test_train_continuous(self):
        if self.irl_continuous is None:
            return
        states = np.zeros(
            shape=(self.batch_size, self.continuous_env.observation_space.low.size),
            dtype=np.float32)
        actions = np.zeros(
            shape=(self.batch_size, self.continuous_env.action_space.low.size),
            dtype=np.float32)
        self.irl_continuous.train(
            agent_states=states,
            agent_acts=actions,
            agent_next_states=states,
            expert_states=states,
            expert_acts=actions,
            expert_next_states=states)


if __name__ == '__main__':
    unittest.main()
