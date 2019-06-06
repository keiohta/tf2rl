import argparse
import numpy as np
import gym
import roboschool

from tf2rl.algos.apex_multienv import apex_argument, run
from tf2rl.algos.ddpg import DDPG
from tf2rl.misc.target_update_ops import update_target_variables


if __name__ == '__main__':
    parser = apex_argument()
    parser.add_argument('--env-name', type=str,
                        default="RoboschoolAtlasForwardWalk-v1")
    args = parser.parse_args()

    # Prepare env and policy function
    def env_fn():
        return gym.make(args.env_name)

    def policy_fn(env, name, memory_capacity=int(1e6), gpu=-1):
        return DDPG(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            gpu=gpu,
            name=name,
            sigma=0.1,
            batch_size=100,
            lr_actor=0.001,
            lr_critic=0.001,
            actor_units=[400, 300],
            critic_units=[400, 300],
            memory_capacity=memory_capacity)

    def get_weights_fn(policy):
        return [policy.actor.weights,
                policy.critic.weights,
                policy.critic_target.weights]

    def set_weights_fn(policy, weights):
        actor_weights, critic_weights, critic_target_weights = weights
        update_target_variables(
            policy.actor.weights, actor_weights, tau=1.)
        update_target_variables(
            policy.critic.weights, critic_weights, tau=1.)
        update_target_variables(
            policy.critic_target.weights, critic_target_weights, tau=1.)

    run(args, env_fn, policy_fn, get_weights_fn, set_weights_fn)
