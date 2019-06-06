import time
import numpy as np
import tensorflow as tf
import argparse

import multiprocessing
from multiprocessing import Process, Queue, Value, Event, Lock
from multiprocessing.managers import SyncManager

from cpprb.experimental import ReplayBuffer, PrioritizedReplayBuffer

from tf2rl.envs.multi_thread_env import MultiThreadEnv
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.get_replay_buffer import get_default_rb_dict

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


def explorer(global_rb, queue, trained_steps, is_training_done,
             lock, env_fn, policy_fn, set_weights_fn, noise_level,
             n_env=64, n_thread=4, buffer_size=1024, episode_max_steps=1000, gpu=0):
    """
    Collect transitions and store them to prioritized replay buffer.
    Args:
        global_rb:
            Prioritized replay buffer sharing with multiple explorers and only one learner.
            This object is shared over processes, so it must be locked when trying to
            operate something with `lock` object.
        queue:
            A FIFO shared with the learner to get latest network parameters.
            This is process safe, so you don't need to lock process when use this.
        trained_steps:
            Number of steps to apply gradients.
        is_training_done:
            multiprocessing.Event object to share the status of training.
        lock:
            multiprocessing.Lock to lock other processes. You must release after process is done.
        env_fn:
            Method object to generate an environment.
        policy_fn:
            Method object to generate an explorer.
        buffer_size:
            Size of local buffer. If it is filled with transitions, add them to `global_rb`
        episode_max_steps:
            Maximum number of steps of an episode.
    """
    if n_env > 1:
        envs = MultiThreadEnv(
            env_fn=env_fn, batch_size=n_env, thread_pool=n_thread,
            max_episode_steps=episode_max_steps)
        env = envs._sample_env
    else:
        env = env_fn()

    policy = policy_fn(
        env=env, name="Explorer", memory_capacity=global_rb.get_buffer_size(),
        noise_level=noise_level, gpu=gpu)

    kwargs = get_default_rb_dict(buffer_size, env)
    if n_env > 1:
        kwargs["env_dict"]["priorities"] = {}
    local_rb = ReplayBuffer(**kwargs)

    if n_env == 1:
        s = env.reset()
        episode_steps = 0
        total_reward = 0.
        total_rewards = []
    else:
        obses = envs.py_reset()
    start = time.time()
    n_sample, n_sample_old = 0, 0

    while not is_training_done.is_set():
        if n_env == 1:
            n_sample += 1
            episode_steps += 1
            a = policy.get_action(s)
            s_, r, done, _ = env.step(a)
            done_flag = done
            if episode_steps == env._max_episode_steps:
                done_flag = False
            total_reward += r
            local_rb.add(obs=s, act=a, rew=r, next_obs=s_, done=done_flag)

            s = s_
            if done or episode_steps == episode_max_steps:
                s = env.reset()
                total_rewards.append(total_reward)
                total_reward = 0
                episode_steps = 0
        else:
            n_sample += n_env
            obses = envs.py_observation()
            actions = policy.get_action(obses, tensor=True)
            next_obses, rewards, dones, _ = envs.step(actions)
            td_errors = policy.compute_td_error(
                states=obses, actions=actions, next_states=next_obses,
                rewards=rewards, dones=dones)
            local_rb.add(obs=obses, act=actions, next_obs=next_obses,
                     rew=rewards, done=dones,
                     priorities=np.abs(td_errors+1e-6))

        # Periodically copy weights of explorer
        if not queue.empty():
            set_weights_fn(policy, queue.get())

        # Add collected experiences to global replay buffer
        if local_rb.get_stored_size() == buffer_size:
            samples = local_rb.sample(local_rb.get_stored_size())
            if n_env > 1:
                priorities=np.squeeze(samples["priorities"])
            else:
                td_errors = policy.compute_td_error(
                    states=samples["obs"], actions=samples["act"],
                    next_states=samples["next_obs"], rewards=samples["rew"],
                    dones=samples["done"])
                priorities = np.abs(td_errors) + 1e-6
            lock.acquire()
            global_rb.add(
                obs=samples["obs"], act=samples["act"], rew=samples["rew"],
                next_obs=samples["next_obs"], done=samples["done"],
                priorities=priorities)
            lock.release()
            local_rb.clear()

            msg = "Grad: {0: 6d}\t".format(trained_steps.value)
            msg += "Samples: {0: 7d}\t".format(n_sample)
            msg += "TDErr: {0:.5f}\t".format(np.average(priorities))
            if n_env == 1:
                ave_rew = 0 if len(total_rewards) == 0 else sum(total_rewards) / len(total_rewards)
                msg += "AveEpiRew: {0:.3f}\t".format(ave_rew)
                total_rewards = []
            msg += "FPS: {0:.2f}".format((n_sample - n_sample_old) / (time.time() - start))
            print(msg, flush=True)

            start = time.time()
            n_sample_old = n_sample


def learner(global_rb, trained_steps, is_training_done,
            lock, env, policy_fn, get_weights_fn,
            n_training, update_freq, evaluation_freq, gpu, queues):
    """
    Collect transitions and store them to prioritized replay buffer.
    Args:
        global_rb:
            Prioritized replay buffer sharing with multiple explorers and only one learner.
            This object is shared over processes, so it must be locked when trying to
            operate something with `lock` object.
        trained_steps:
            Number of times to apply gradients.
        is_training_done:
            multiprocessing.Event object to share if training is done or not.
        lock:
            multiprocessing.Lock to lock other processes.
            It must be released after process is done.
        env_fn:
            Environment.
        policy_fn:
            Method object to generate an explorer.
        n_training:
            Maximum number of times to apply gradients. If number of applying gradients
            is over this value, training will be done by setting `is_training_done` to `True`
        update_freq:
            Frequency to update parameters, i.e., put network parameters to `queues`
        queues:
            FIFOs shared with explorers to send latest network parameters.
    """
    policy = policy_fn(env, "Learner", global_rb.get_buffer_size(), gpu=gpu)

    output_dir = prepare_output_dir(
        args=None, user_specified_dir="./results", suffix="learner")
    writer = tf.contrib.summary.create_file_writer(output_dir)
    writer.set_as_default()
    tf.contrib.summary.initialize()
    total_steps = tf.train.create_global_step()

    # Wait until explorers collect transitions
    while not is_training_done.is_set() and global_rb.get_stored_size() == 0:
        continue

    start_time = time.time()
    while not is_training_done.is_set():
        with tf.contrib.summary.record_summaries_every_n_global_steps(1000):
            trained_steps.value += 1
            total_steps.assign(trained_steps.value)
            lock.acquire()
            samples = global_rb.sample(policy.batch_size)
            with tf.contrib.summary.always_record_summaries():
                td_errors = policy.train(
                    samples["obs"], samples["act"], samples["next_obs"],
                    samples["rew"], samples["done"], samples["weights"])
                writer.flush()
            global_rb.update_priorities(
                samples["indexes"], np.abs(td_errors)+1e-6)
            lock.release()

            # Put updated weights to queue
            if trained_steps.value % update_freq == 0:
                weights = get_weights_fn(policy)
                for i in range(len(queues) - 1):
                    queues[i].put(weights)
                with tf.contrib.summary.always_record_summaries():
                    fps = update_freq / (time.time() - start_time)
                    tf.contrib.summary.scalar(name="FPS", tensor=fps, family="loss")
                    print("Update weights for explorer. {0:.2f} FPS for GRAD. Learned {1:.2f} steps".format(
                        fps, trained_steps.value), flush=True)
                start_time = time.time()

            # Periodically do evaluation
            if trained_steps.value % evaluation_freq == 0:
                queues[-1].put(get_weights_fn(policy))
                queues[-1].put(int(total_steps))

        if trained_steps.value >= n_training:
            is_training_done.set()


def apex_argument(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--max-batch', type=int, default=1e7,
                        help='number of times to apply batch update')
    parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                        help='Maximum steps in an episode')
    parser.add_argument('--param-update-freq', type=int, default=1e2,
                        help='frequency to update parameter')
    parser.add_argument('--test-freq', type=int, default=1e3,
                        help='Frequency to evaluate policy')
    parser.add_argument('--n-explorer', type=int, default=None,
                        help='number of explorers to distribute. if None, use maximum number')
    parser.add_argument('--replay-buffer-size', type=int, default=1e6,
                        help='size of replay buffer')
    parser.add_argument('--local-buffer-size', type=int, default=1e4,
                        help='size of local replay buffer for explorer')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu-explorer', type=int, default=0)
    parser.add_argument('--gpu-learner', type=int, default=0)
    parser.add_argument('--gpu-evaluator', type=int, default=0)
    # Multi Env setting
    parser.add_argument('--n-env', type=int, default=1,
                        help='Number of environments')
    parser.add_argument('--n-thread', type=int, default=4,
                        help='Number of thread pool')
    return parser


def evaluator(is_training_done, env, policy_fn, set_weights_fn, queue, gpu,
              n_evaluation=10, episode_max_steps=1000, show_test_progress=False):
    output_dir = prepare_output_dir(
        args=None, user_specified_dir="./results", suffix="_evaluator")
    writer = tf.contrib.summary.create_file_writer(
        output_dir, filename_suffix="_evaluation")
    writer.set_as_default()
    tf.contrib.summary.initialize()

    policy = policy_fn(env, "Learner", gpu=gpu)
    total_steps = tf.train.create_global_step()

    while not is_training_done.is_set():
        n_evaluated_episode = 0
        # Wait until a new weights comes
        if queue.empty():
            continue
        else:
            set_weights_fn(policy, queue.get())
            total_steps.assign(queue.get())
            avg_test_return = 0.
            for i in range(n_evaluation):
                n_evaluated_episode += 1
                episode_return = 0.
                obs = env.reset()
                done = False
                for _ in range(episode_max_steps):
                    action = policy.get_action(obs, test=True)
                    next_obs, reward, done, _ = env.step(action)
                    if show_test_progress:
                        env.render()
                    episode_return += reward
                    obs = next_obs
                    if done:
                        break
                avg_test_return += episode_return
                # Break if a new weights comes
                if not queue.empty():
                    break
            avg_test_return /= n_evaluated_episode
            print("Evaluation: {} over {} run".format(
                avg_test_return, n_evaluated_episode), flush=True)
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(name="AverageTestReturn",
                                          tensor=avg_test_return, family="loss")
                writer.flush()


def prepare_experiment(env, args):
    # Manager to share PER between a learner and explorers
    SyncManager.register('PrioritizedReplayBuffer',
                         PrioritizedReplayBuffer)
    manager = SyncManager()
    manager.start()

    kwargs = get_default_rb_dict(args.replay_buffer_size, env)
    global_rb = manager.PrioritizedReplayBuffer(**kwargs)

    # queues to share network parameters between a learner and explorers
    n_queue = 1 if args.n_env > 1 else args.n_explorer
    n_queue += 1  # for evaluation
    queues = [manager.Queue() for _ in range(n_queue)]

    # Event object to share training status. if event is set True, all exolorers stop sampling transitions
    is_training_done = Event()

    # Lock
    lock = manager.Lock()

    # Shared memory objects to count number of samples and applied gradients
    trained_steps = Value('i', 0)

    return global_rb, queues, is_training_done, lock, trained_steps


def run(args, env_fn, policy_fn, get_weights_fn, set_weights_fn):
    if args.n_env > 1:
        args.n_explorer = 1
    elif args.n_explorer is None:
        args.n_explorer = multiprocessing.cpu_count() - 1
    assert args.n_explorer > 0, "[error] number of explorers must be positive integer"

    env = env_fn()

    global_rb, queues, is_training_done, lock, trained_steps = \
        prepare_experiment(env, args)

    noise = 0.3
    tasks = []

    # Add explorers
    if args.n_env > 1:
        tasks.append(Process(
            target=explorer,
            args=[global_rb, queues[0], trained_steps, is_training_done,
                  lock, env_fn, policy_fn, set_weights_fn, noise,
                  args.n_env, args.n_thread, args.local_buffer_size,
                  args.episode_max_steps, args.gpu_explorer]))
    else:
        for i in range(args.n_explorer):
            tasks.append(Process(
                target=explorer,
                args=[global_rb, queues[i], trained_steps, is_training_done,
                      lock, env_fn, policy_fn, set_weights_fn, noise,
                      args.n_env, args.n_thread, args.local_buffer_size,
                      args.episode_max_steps, args.gpu_explorer]))

    # Add learner
    tasks.append(Process(
        target=learner,
        args=[global_rb, trained_steps, is_training_done,
                lock, env_fn(), policy_fn, get_weights_fn,
                args.max_batch, args.param_update_freq,
                args.test_freq, args.gpu_learner, queues]))

    # Add evaluator
    tasks.append(Process(
        target=evaluator,
        args=[is_training_done, env_fn(), policy_fn, set_weights_fn,
              queues[-1], args.gpu_evaluator]))

    for task in tasks:
        task.start()
    for task in tasks:
        task.join()
