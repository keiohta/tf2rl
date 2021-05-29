import dmc2gym

from tf2rl.algos.curl_sac import CURLSAC
from tf2rl.envs.dmc_wrapper import DMCWrapper
from tf2rl.experiments.trainer import Trainer


def main():
    dm_envs = {
        'finger': ['finger', 'spin', 2],
        'cartpole': ['cartpole', 'balance', 4],
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
    parser = CURLSAC.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="cartpole", choices=dm_envs.keys())
    parser.add_argument('--seed', type=int, default=1)
    parser.set_defaults(batch_size=256)
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(max_steps=3e6)
    parser.set_defaults(save_summary_interval=100)
    args = parser.parse_args()

    domain_name, task_name, action_repeat = dm_envs[args.env_name]
    original_obs_shape = (100, 100, 9)
    input_obs_shape = (84, 84, 9)

    def make_env():
        return DMCWrapper(
            dmc2gym.make(
                domain_name=domain_name,
                task_name=task_name,
                seed=args.seed,
                visualize_reward=False,
                from_pixels=True,
                height=100,
                width=100,
                frame_skip=action_repeat,
                channels_first=False),
            obs_shape=original_obs_shape,
            k=3,
            channel_first=False)

    env = make_env()
    test_env = make_env()

    # see Table 3 of CURL paper
    lr_sac = lr_curl = 2e-4 if args.env_name == "cheetah" else 1e-3

    policy = CURLSAC(
        obs_shape=input_obs_shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=int(1e5),
        n_warmup=int(1e3),
        max_action=env.action_space.high[0],
        batch_size=512,
        actor_units=(1024, 1024),
        critic_units=(1024, 1024),
        lr_sac=lr_sac,
        lr_curl=lr_curl,
        lr_alpha=1e-4,
        tau=0.01,
        init_temperature=0.1,
        auto_alpha=True,
        stop_q_grad=args.stop_q_grad)

    trainer = Trainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()


if __name__ == "__main__":
    main()
