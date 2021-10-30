import unittest

from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer
from tf2rl.envs.utils import make


class TestTrainer(unittest.TestCase):
    def test_empty_args(self):
        """
        Test empty args {}
        """
        env = make("Pendulum-v0")
        test_env = make("Pendulum-v0")
        policy = DDPG(state_shape=env.observation_space.shape,
                      action_dim=env.action_space.high.size,
                      gpu=-1,
                      memory_capacity=1000,
                      max_action=env.action_space.high[0],
                      batch_size=32,
                      n_warmup=10)
        Trainer(policy, env, {}, test_env=test_env)

    def test_with_args(self):
        """
        Test with args
        """
        max_steps = 400
        env = make("Pendulum-v0")
        test_env = make("Pendulum-v0")
        policy = DDPG(state_shape=env.observation_space.shape,
                      action_dim=env.action_space.high.size,
                      gpu=-1,
                      memory_capacity=1000,
                      max_action=env.action_space.high[0],
                      batch_size=32,
                      n_warmup=10)
        trainer = Trainer(policy, env, {"max_steps": max_steps}, test_env=test_env)
        self.assertEqual(trainer._max_steps, max_steps)

    def test_invalid_args(self):
        """
        Test with invalid args
        """
        env = make("Pendulum-v0")
        test_env = make("Pendulum-v0")
        policy = DDPG(state_shape=env.observation_space.shape,
                      action_dim=env.action_space.high.size,
                      gpu=-1,
                      memory_capacity=1000,
                      max_action=env.action_space.high[0],
                      batch_size=32,
                      n_warmup=10)
        with self.assertRaises(ValueError):
            Trainer(policy, env, {"NOT_EXISTING_OPTIONS": 1}, test_env=test_env)


if __name__ == "__main__":
    unittest.main()
