import gym

import numpy as np

from tf2rl.algos.dqn import DQN
from tf2rl.experiments.trainer import Trainer
from tf2rl.envs.utils import get_shape, is_discrete


class DummyDiscreteEnv(gym.Env):
    def __init__(self):
        self.obs_dim, self.act_dim = 3, 2
        self.observation_space = gym.spaces.discrete.Discrete(self.obs_dim)
        self.action_space = gym.spaces.discrete.Discrete(self.act_dim)

    def _get_obs(self):
        return np.random.randint(self.obs_dim)

    def reset(self):
        return self._get_obs()

    def step(self, action):
        assert action < self.act_dim
        return self._get_obs(), 0, False, {}


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = DQN.get_argument(parser)
    parser.set_defaults(test_interval=2000)
    parser.set_defaults(max_steps=100000)
    parser.set_defaults(gpu=-1)
    parser.set_defaults(n_warmup=500)
    parser.set_defaults(batch_size=32)
    parser.set_defaults(memory_capacity=int(1e4))
    args = parser.parse_args()

    env = DummyDiscreteEnv()

    policy = DQN(
        state_shape=get_shape(env.observation_space),
        action_dim=env.action_space.n,
        discrete_input=is_discrete(env.observation_space),
        enable_double_dqn=args.enable_double_dqn,
        enable_dueling_dqn=args.enable_dueling_dqn,
        enable_noisy_dqn=args.enable_noisy_dqn,
        enable_categorical_dqn=args.enable_categorical_dqn,
        target_replace_interval=300,
        discount=0.99,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)

    state = np.zeros(shape=(env.observation_space.n))
    state[env.reset()] = 1

    trainer = Trainer(policy, env, args)
    trainer()
