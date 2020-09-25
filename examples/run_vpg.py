import gym

from tf2rl.algos.vpg import VPG
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim


if __name__ == '__main__':
    parser = OnPolicyTrainer.get_argument()
    parser = VPG.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="Pendulum-v0")
    parser.set_defaults(test_interval=10240)
    parser.set_defaults(max_steps=int(1e7))
    parser.set_defaults(horizon=512)
    parser.set_defaults(batch_size=32)
    parser.set_defaults(gpu=-1)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = VPG(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        max_action=None if is_discrete(
            env.action_space) else env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=(32, 32),
        critic_units=(32, 32),
        discount=0.9,
        horizon=args.horizon,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)
    trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
    trainer()
