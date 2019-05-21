import argparse
import numpy as np
import gym

import multiprocessing
from multiprocessing import Process

from tf2rl.algos.apex import apex_argument, explorer, learner, prepare_experiment
from tf2rl.algos.td3 import TD3


if __name__ == '__main__':
    parser = apex_argument()
    args = parser.parse_args()

    if args.n_explorer is None:
        n_explorer = multiprocessing.cpu_count() - 1
    else:
        n_explorer = args.n_explorer
    assert n_explorer > 0, "[error] number of explorers must be positive integer"

    # Prepare env and policy function
    def env_fn():
        return gym.make('HalfCheetah-v2')

    def policy_fn(env, name, memory_capacity=int(1e6),
                  gpu=-1, sigma=0.1):
        return TD3(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            gpu=gpu,
            name=name,
            sigma=sigma,
            batch_size=100,
            memory_capacity=memory_capacity)

    env = env_fn()

    global_rb, queues, is_training_done, lock, \
        trained_steps, n_transition = \
        prepare_experiment(n_explorer, env, args)

    tasks = []
    noises = np.linspace(0.01, 0.3, n_explorer)
    # Add explorers
    for i in range(n_explorer):
        env = env_fn()
        tasks.append(Process(
            target=explorer,
            args=[global_rb, queues[i], trained_steps, n_transition, is_training_done, lock, 
                  env, policy_fn, noises[i], args.local_buffer_size]))

    # Add learner
    tasks.append(Process(
        target=learner,
        args=[global_rb, trained_steps, is_training_done, lock, env_fn(), policy_fn,
              args.max_batch, args.param_update_freq, *queues]))

    for task in tasks:
        task.start()
    for task in tasks:
        task.join()
