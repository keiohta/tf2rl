import unittest
import gym
import tensorflow as tf

from tf2rl.algos.dqn import DQN


class DQNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("CartPole-v0")

    def test__init__(self):
        DQN(state_dim=self.env.observation_space.low.size,
            action_dim=self.env.action_space.n,
            gpu=-1)

    def test_get_action(self):
        agent = DQN(
            state_dim=self.env.observation_space.low.size,
            action_dim=self.env.action_space.n,
            gpu=-1)
        state = self.env.reset()
        agent.get_action(state, test=False)
        agent.get_action(state, test=True)

    def test_train(self):
        agent = DQN(
            state_dim=self.env.observation_space.low.size,
            action_dim=self.env.action_space.n,
            memory_capacity=100,
            gpu=-1)
        from cpprb import ReplayBuffer
        replay_buffer = ReplayBuffer(
            obs_dim=self.env.observation_space.low.size,
            act_dim=1,
            size=agent.memory_capacity)

        obs = self.env.reset()
        for _ in range(100):
            action = agent.get_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)
            if done:
                next_obs = self.env.reset()
            obs = next_obs

        for _ in range(100):
            agent.train(replay_buffer)


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.enable_eager_execution(config=config)
    unittest.main()
