from tf2rl.algos.curl_sac import CURL
from tf2rl.experiments.trainer import Trainer


from examples.run_sacae import dm_envs, make_env, get_common_dmc_kwargs

def main():
    parser = Trainer.get_argument()
    parser = CURL.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="cartpole", choices=dm_envs.keys())
    parser.add_argument('--seed', type=int, default=1)
    parser.set_defaults(save_summary_interval=50)
    parser.set_defaults(memory_capacity=int(1e5))
    args = parser.parse_args()

    original_obs_shape = (100, 100, 9)
    input_obs_shape = (84, 84, 9)

    env = make_env(args, original_obs_shape[0], original_obs_shape[1], original_obs_shape)
    test_env = make_env(args, original_obs_shape[0], original_obs_shape[1], original_obs_shape)

    # see Table 3 of CURL paper
    dmc_kwargs = get_common_dmc_kwargs(args, env)

    policy = CURL(
        obs_shape=input_obs_shape,
        **dmc_kwargs)

    trainer = Trainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()


if __name__ == "__main__":
    main()
