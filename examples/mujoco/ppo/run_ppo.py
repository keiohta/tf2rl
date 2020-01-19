import gym

from tf2rl.algos.ppo import PPO
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import get_act_dim


if __name__ == '__main__':
    parser = OnPolicyTrainer.get_argument()
    parser = PPO.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="HalfCheetah-v2")
    parser.set_defaults(gpu=-1)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    policy = PPO(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=False,
        max_action=env.action_space.high[0],
        batch_size=64,
        actor_units=(64, 64),
        critic_units=(64, 64),
        n_epoch=10,
        lr_actor=3e-4,
        lr_critic=3e-4,
        hidden_activation_actor="tanh",
        hidden_activation_critic="tanh",
        discount=0.995,
        lam=0.97,
        entropy_coef=0.,
        horizon=2048,
        normalize_adv=True,
        enable_gae=True,
        gpu=args.gpu)

    args.test_interval = policy.horizon * 10
    args.save_summary_interval = policy.horizon * 10
    args.max_steps = int(2e6)
    args.normalize_obs = True

    trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
    trainer()
