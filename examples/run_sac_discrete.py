import gym

from tf2rl.algos.sac_discrete import SACDiscrete
from tf2rl.experiments.trainer import Trainer
from tf2rl.envs.utils import is_atari_env
from tf2rl.envs.atari_wrapper import wrap_dqn
from tf2rl.networks.atari_model import AtariQFunc, AtariCategoricalActor


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = SACDiscrete.get_argument(parser)
    parser.set_defaults(test_interval=2000)
    parser.set_defaults(max_steps=100000)
    parser.set_defaults(gpu=-1)
    parser.set_defaults(n_warmup=500)
    parser.set_defaults(batch_size=32)
    parser.set_defaults(memory_capacity=int(1e4))
    parser.add_argument('--env-name', type=str,
                        default="SpaceInvadersNoFrameskip-v4")
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    if is_atari_env(env):
        # Parameters come from Appendix.B in original paper.
        # See https://arxiv.org/abs/1910.07207
        parser.set_defaults(episode_max_steps=108000)
        parser.set_defaults(test_interval=int(1e5))
        parser.set_defaults(show_test_images=True)
        parser.set_defaults(max_steps=int(1e9))
        parser.set_defaults(target_update_interval=8000)
        parser.set_defaults(n_warmup=int(2e4))
        args = parser.parse_args()
        if args.gpu == -1:
            print("Are you sure you're trying to solve Atari without GPU?")

        env = wrap_dqn(env, wrap_ndarray=True)
        test_env = wrap_dqn(test_env, wrap_ndarray=True, reward_clipping=False)
        policy = SACDiscrete(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.n,
            discount=0.99,
            critic_fn=AtariQFunc,
            actor_fn=AtariCategoricalActor,
            lr=3e-4,
            memory_capacity=args.memory_capacity,
            batch_size=64,
            n_warmup=args.n_warmup,
            update_interval=4,
            target_update_interval=args.target_update_interval,
            auto_alpha=args.auto_alpha,
            gpu=args.gpu)
    else:
        policy = SACDiscrete(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.n,
            discount=0.99,
            memory_capacity=args.memory_capacity,
            batch_size=args.batch_size,
            n_warmup=args.n_warmup,
            target_update_interval=args.target_update_interval,
            auto_alpha=args.auto_alpha,
            gpu=args.gpu)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
