import argparse
import numpy as np
import gym
import roboschool

import multiprocessing
from multiprocessing import Process

from tf2rl.algos.apex import apex_argument, explorer, learner, prepare_experiment
from tf2rl.algos.ddpg import DDPG
from tf2rl.misc.target_update_ops import update_target_variables


if __name__ == '__main__':
    parser = apex_argument()
    parser.add_argument('--env-name', type=str,
                        default="RoboschoolAtlasForwardWalk-v1")
    args = parser.parse_args()

    if args.n_explorer is None:
        n_explorer = multiprocessing.cpu_count() - 1
    else:
        n_explorer = args.n_explorer
    assert n_explorer > 0, "[error] number of explorers must be positive integer"

    # Prepare env and policy function
    def env_fn():
        return gym.make(args.env_name)

    def policy_fn(env, name, memory_capacity=int(1e6),
                  gpu=-1, sigma=0.3):
        return DDPG(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            gpu=gpu,
            name=name,
            sigma=sigma,
            batch_size=100,
            lr_actor=0.0001,
            lr_critic=0.0001,
            actor_units=[300, 200],
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

    env = env_fn()

    global_rb, queues, is_training_done, lock, \
        trained_steps, n_transition = \
        prepare_experiment(n_explorer, env, args)

    tasks = []
    noise = 0.3
    # Add explorers
    for i in range(n_explorer):
        env = env_fn()
        tasks.append(Process(
            target=explorer,
            args=[global_rb, queues[i], trained_steps, n_transition,
                  is_training_done, lock, env, policy_fn, set_weights_fn,
                  noise, args.local_buffer_size]))

    # Add learner
    tasks.append(Process(
        target=learner,
        args=[global_rb, trained_steps, is_training_done, lock, env_fn(),
              policy_fn, get_weights_fn,
              args.max_batch, args.param_update_freq, *queues]))

    for task in tasks:
        task.start()
    for task in tasks:
        task.join()
