import gym

from tf2rl.algos.categorical_dqn import CategoricalDQN, CategoricalQFunc
from tf2rl.trainer.trainer import Trainer


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = CategoricalDQN.get_argument(parser)
    parser.set_defaults(test_interval=2000)
    parser.set_defaults(max_steps=int(5e5))
    parser.set_defaults(gpu=-1)
    args = parser.parse_args()

    env = gym.make("CartPole-v0")
    test_env = gym.make("CartPole-v0")
    policy = CategoricalDQN(
        enable_double_dqn=args.enable_double_dqn,
        enable_dueling_dqn=args.enable_dueling_dqn,
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        q_func=CategoricalQFunc,
        n_warmup=500,
        target_replace_interval=300,
        batch_size=32,
        memory_capacity=10000,
        discount=0.99,
        gpu=args.gpu)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
