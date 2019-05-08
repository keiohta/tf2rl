import gym

from tf2rl.algos.dqn import DQN
from tf2rl.trainer.trainer import Trainer


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser.set_defaults()
    parser.set_defaults(test_interval=2000)
    parser.set_defaults(gpu=-1)
    args = parser.parse_args()

    env = gym.make("CartPole-v0")
    test_env = gym.make("CartPole-v0")
    policy = DQN(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        n_warmup=500,
        target_replace_interval=100,
        batch_size=32,
        memory_capacity=10000,
        discount=0.9,
        gpu=args.gpu)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
