import gym

from tf2rl.algos.sac_discrete import SACDiscrete
from tf2rl.experiments.trainer import Trainer


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = SACDiscrete.get_argument(parser)
    parser.set_defaults(test_interval=2000)
    parser.set_defaults(max_steps=100000)
    parser.set_defaults(gpu=-1)
    parser.set_defaults(n_warmup=500)
    parser.set_defaults(batch_size=32)
    parser.set_defaults(memory_capacity=int(1e4))
    args = parser.parse_args()

    env = gym.make("CartPole-v0")
    test_env = gym.make("CartPole-v0")
    policy = SACDiscrete(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        discount=0.99,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
