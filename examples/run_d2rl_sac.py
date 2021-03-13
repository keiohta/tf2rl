import gym

from tf2rl.algos.d2rl_sac import D2RLSAC
from tf2rl.experiments.trainer import Trainer


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = D2RLSAC.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    parser.set_defaults(batch_size=256)
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(max_steps=3e6)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = D2RLSAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        actor_units=(256, 256, 256, 256),
        critic_units=(256, 256, 256, 256),
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha)
    trainer = Trainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()
