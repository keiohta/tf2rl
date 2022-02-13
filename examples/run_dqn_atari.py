from tf2rl.algos.dqn import DQN
from tf2rl.envs.atari_wrapper import wrap_dqn
from tf2rl.experiments.trainer import Trainer
from tf2rl.networks.atari_model import AtariQFunc as QFunc
from tf2rl.envs.utils import make

if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = DQN.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="SpaceInvadersNoFrameskip-v4")
    parser.set_defaults(episode_max_steps=108000)
    parser.set_defaults(test_interval=10000)
    parser.set_defaults(max_steps=int(1e9))
    parser.set_defaults(save_model_interval=500000)
    parser.set_defaults(gpu=0)
    parser.set_defaults(show_test_images=True)
    parser.set_defaults(memory_capacity=int(1e6))
    args = parser.parse_args()

    env = wrap_dqn(make(args.env_name))
    test_env = wrap_dqn(make(args.env_name), reward_clipping=False)
    # Following parameters are equivalent to DeepMind DQN paper
    # https://www.nature.com/articles/nature14236
    policy = DQN(
        enable_double_dqn=args.enable_double_dqn,
        enable_dueling_dqn=args.enable_dueling_dqn,
        enable_noisy_dqn=args.enable_noisy_dqn,
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        lr=0.0000625,  # This value is from Rainbow
        adam_eps=1.5e-4,  # This value is from Rainbow
        n_warmup=50000,
        target_replace_interval=10000,
        batch_size=32,
        memory_capacity=args.memory_capacity,
        discount=0.99,
        epsilon=1.,
        epsilon_min=0.1,
        epsilon_decay_step=int(1e6),
        update_interval=4,
        q_func=QFunc,
        gpu=args.gpu)
    trainer = Trainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()
