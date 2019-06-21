import gym

from tf2rl.algos.vpg import VPG
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim


if __name__ == '__main__':
    parser = OnPolicyTrainer.get_argument()
    parser = VPG.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="Pendulum-v0")
    parser.add_argument('--normalize-adv', action='store_true')
    parser.add_argument('--enable-gae', action='store_true')
    parser.set_defaults(test_interval=5000)
    parser.set_defaults(max_steps=int(1e6))
    parser.set_defaults(horizon=512)
    parser.set_defaults(gpu=-1)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = VPG(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=[32, 32],
        critic_units=[32, 32],
        discount=0.9,
        horizon=args.horizon,
        fix_std=True,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)
    trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
    trainer()
