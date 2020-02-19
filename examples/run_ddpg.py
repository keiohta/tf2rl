import gym

from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = DDPG.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
