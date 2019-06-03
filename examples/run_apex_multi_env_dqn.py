import argparse
import numpy as np
import gym

from tf2rl.algos.apex_multienv import apex_argument, run
from tf2rl.algos.dqn import DQN
from tf2rl.misc.target_update_ops import update_target_variables


if __name__ == '__main__':
    parser = apex_argument()
    parser = DQN.get_argument(parser)
    parser.add_argument('--atari', action='store_true')
    parser.add_argument('--env-name', type=str,
                        default="SpaceInvadersNoFrameskip-v4")
    args = parser.parse_args()

    # Prepare env and policy function
    def env_fn():
        if args.atari:
            return gym.make(args.env_name)
        else:
            return gym.make("CartPole-v0")

    def policy_fn(env, name, memory_capacity=int(1e6), gpu=-1):
        return DQN(
            name=name,
            enable_double_dqn=args.enable_double_dqn,
            enable_dueling_dqn=args.enable_dueling_dqn,
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.n,
            n_warmup=500,
            target_replace_interval=300,
            batch_size=32,
            memory_capacity=memory_capacity,
            discount=0.99,
            gpu=gpu)

    def get_weights_fn(policy):
        return [policy.q_func.weights,
                policy.q_func_target.weights]

    def set_weights_fn(policy, weights):
        q_func_weights, qfunc_target_weights = weights
        update_target_variables(
            policy.q_func.weights, q_func_weights, tau=1.)
        update_target_variables(
            policy.q_func_target.weights, qfunc_target_weights, tau=1.)

    run(args, env_fn, policy_fn, get_weights_fn, set_weights_fn)
