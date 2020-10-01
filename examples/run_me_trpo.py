import gym

from tf2rl.algos.ppo import PPO
from tf2rl.envs.utils import is_discrete, get_act_dim
from tf2rl.experiments.me_trpo_trainer import MeTrpoTrainer
from examples.run_mpc import reward_fn_pendulum


if __name__ == "__main__":
    parser = MeTrpoTrainer.get_argument()
    parser = PPO.get_argument(parser)
    parser.set_defaults(episode_max_steps=100)
    parser.set_defaults(n_collect_steps=2048)
    parser.set_defaults(n_generate_steps=2048)
    args = parser.parse_args()

    args.n_generate_steps = args.horizon

    env = gym.make("Pendulum-v0")
    test_env = gym.make("Pendulum-v0")

    policy = PPO(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        max_action=None if is_discrete(
            env.action_space) else env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=(32, 32),
        critic_units=(32, 32),
        n_epoch=10,
        lr_actor=3e-4,
        lr_critic=3e-4,
        hidden_activation_actor="tanh",
        hidden_activation_critic="tanh",
        discount=0.9,
        lam=0.95,
        entropy_coef=0.,
        horizon=args.horizon,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)

    trainer = MeTrpoTrainer(policy, env, args, reward_fn=reward_fn_pendulum, test_env=test_env)
    trainer()
