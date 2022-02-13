import dmc2gym

from tf2rl.algos.sac_ae import SACAE
from tf2rl.envs.dmc_wrapper import DMCWrapper
from tf2rl.experiments.trainer import Trainer


def main():
    dm_envs = {
        'finger': ['finger', 'spin', 2],
        'cartpole': ['cartpole', 'swingup', 8],
        'reacher': ['reacher', 'easy', 4],
        'cheetah': ['cheetah', 'run', 4],
        'walker': ['walker', 'walk', 2],
        'ball': ['ball_in_cup', 'catch', 4],
        'humanoid': ['humanoid', 'stand', 4],
        'bring_ball': ['manipulator', 'bring_ball', 4],
        'bring_peg': ['manipulator', 'bring_peg', 4],
        'insert_ball': ['manipulator', 'insert_ball', 4],
        'insert_peg': ['manipulator', 'insert_peg', 4]}

    parser = Trainer.get_argument()
    parser = SACAE.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="cartpole", choices=dm_envs.keys())
    parser.add_argument('--seed', type=int, default=1)
    parser.set_defaults(save_summary_interval=50)
    args = parser.parse_args()

    domain_name, task_name, action_repeat = dm_envs[args.env_name]
    input_obs_shape = (84, 84, 9)

    def make_env():
        return DMCWrapper(
            dmc2gym.make(
                domain_name=domain_name,
                task_name=task_name,
                seed=args.seed,
                visualize_reward=False,
                from_pixels=True,
                height=input_obs_shape[0],
                width=input_obs_shape[1],
                frame_skip=action_repeat,
                channels_first=False),
            obs_shape=input_obs_shape,
            k=3,
            channel_first=False)

    env = make_env()
    test_env = make_env()

    # see Table 3 of CURL paper
    lr_sac = lr_ae = 2e-4 if args.env_name == "cheetah" else 1e-3

    policy = SACAE(
        obs_shape=input_obs_shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=int(1e4),
        n_warmup=int(1e3),
        max_action=env.action_space.high[0],
        batch_size=128,
        actor_units=(1024, 1024),
        critic_units=(1024, 1024),
        lr_sac=lr_sac,
        lr_encoder=lr_ae,
        lr_decoder=lr_ae,
        lr_alpha=1e-4,
        tau_critic=0.01,
        init_temperature=0.1,
        auto_alpha=True)

    trainer = Trainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()


if __name__ == "__main__":
    main()
