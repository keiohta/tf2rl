import gym

from tf2rl.algos.dqn import DQN
from tf2rl.trainer.trainer import Trainer


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser.set_defaults()
    args = parser.parse_args()

    env = gym.make("CartPole-v0")
    test_env = gym.make("CartPole-v0")
    policy = DQN(
        state_dim=env.observation_space.low.size,
        action_dim=env.action_space.n,
        n_warmup=500,
        target_replace_interval=100,
        batch_size=64,
        memory_capacity=1000000)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
