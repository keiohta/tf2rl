import unittest
import gym

from tf2rl.envs.utils import is_discrete


class TestUtils(unittest.TestCase):
    def test_is_discrete(self):
        discrete_space = gym.make('CartPole-v0').action_space
        continuous_space = gym.make('Pendulum-v0').action_space
        self.assertTrue(is_discrete(discrete_space))
        self.assertFalse(is_discrete(continuous_space))


if __name__ == '__main__':
    unittest.main()
