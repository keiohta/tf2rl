import gym

from tf2rl.algos.vpg import VPG
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim


if __name__ == '__main__':
    parser = OnPolicyTrainer.get_argument()
    parser = VPG.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="Pendulum-v0")
    parser.set_defaults(test_interval=2000)
    parser.set_defaults(max_steps=int(1e6))
    parser.set_defaults(batch_size=32)
    parser.set_defaults(gpu=-1)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = VPG(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        batch_size=args.batch_size,
        discount=0.9,
        gpu=args.gpu)
    trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
    trainer()
