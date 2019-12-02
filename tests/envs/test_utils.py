import unittest
import gym

from tf2rl.envs.utils import is_discrete, is_atari_env


class TestUtils(unittest.TestCase):
    def test_is_discrete(self):
        discrete_space = gym.make('CartPole-v0').action_space
        continuous_space = gym.make('Pendulum-v0').action_space
        self.assertTrue(is_discrete(discrete_space))
        self.assertFalse(is_discrete(continuous_space))

    @unittest.skip('Skip this because this needs to install mujoco_py and its license')
    def test_is_mujoco_env(self):
        from tf2rl.envs.utils import is_mujoco_env
        self.assertTrue(is_mujoco_env(
            gym.make("HalfCheetah-v2")))
        self.assertFalse(is_mujoco_env(
            gym.make("Pendulum-v0")))

    def test_is_atari_env(self):
        self.assertTrue(is_atari_env(
            gym.make("SpaceInvadersNoFrameskip-v4")))
        self.assertFalse(is_atari_env(
            gym.make("Pendulum-v0")))


if __name__ == '__main__':
    unittest.main()
