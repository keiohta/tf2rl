import gym

from tf2rl.algos.vpg import VPG
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete


if __name__ == '__main__':
    parser = OnPolicyTrainer.get_argument()
    parser.set_defaults(test_interval=2000)
    parser.set_defaults(max_steps=int(5e5))
    parser.set_defaults(gpu=-1)
    args = parser.parse_args()

    env = gym.make("Pendulum-v0")
    test_env = gym.make("Pendulum-v0")
    policy = VPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.low.size,
        is_discrete=is_discrete(env.action_space),
        batch_size=32,
        discount=0.99,
        gpu=args.gpu)
    trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
    trainer()
