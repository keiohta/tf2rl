import time
import numpy as np
import tensorflow as tf
import argparse

import multiprocessing
from multiprocessing import Process, Queue, Value, Event, Lock
from multiprocessing.managers import SyncManager

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.envs.multi_thread_env import MultiThreadEnv
from tf2rl.envs.env_utils import get_act_dim

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


def explorer(global_rb, queue, trained_steps,
             is_training_done, lock, env_fn, policy_fn,
             set_weights_fn, n_env=64, n_thread=4,
             buffer_size=1024, episode_max_steps=1000):
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
        max_transition:
            Maximum number of steps to explorer. Default value is None.
        episode_max_steps:
            Maximum number of steps of an episode.
    """
    envs = MultiThreadEnv(
        env_fn, n_env, n_thread, episode_max_steps)
    policy = policy_fn(
        envs._sample_env, "Explorer", global_rb.get_buffer_size())
    local_rb = ReplayBuffer(
        obs_shape=envs._sample_env.observation_space.shape,
        act_dim=get_act_dim(envs._sample_env),
        size=buffer_size)

    obses = envs.py_reset()
    start = time.time()
    n_sample, n_sample_old = 0, 0
    while not is_training_done.is_set():
        n_sample += n_env
        obses = envs.py_observation()
        actions = policy.get_action(obses, tensor=True)
        next_obses, rewards, dones, _ = envs.step(actions)
        local_rb.add(obs=obses, act=actions, next_obs=next_obses,
                     rew=rewards, done=dones)

        # Periodically copy weights of explorer
        if not queue.empty():
            set_weights_fn(policy, queue.get())

        # Add collected experiences to global replay buffer
        if local_rb.get_stored_size() > buffer_size:
            samples = local_rb.sample(local_rb.get_stored_size())
            lock.acquire()
            global_rb.add(
                obs=samples["obs"], act=samples["act"],
                next_obs=samples["next_obs"], 
                rew=samples["rew"], done=samples["done"],)
                # priorities=np.abs(td_errors)+1e-6)
            lock.release()
            local_rb.clear()
            msg = "Grad: {0: 6d}\t".format(trained_steps.value)
            msg += "Samples: {0: 7d}\t".format(n_transition.value)
            # msg += "TDErr: {0:.5f}\t".format(np.average(np.abs(td_errors).flatten()))
            msg += "FPS: {0:.2f}".format((n_transition.value - sample_at_start) / (time.time() - start))
            print(msg)
            start = time.time()
            n_sample_old = n_sample


def learner(global_rb, trained_steps, is_training_done,
            lock, env, policy_fn, get_weights_fn,
            n_training, update_freq, evaluation_freq, queues):
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
    policy = policy_fn(env, "Learner", global_rb.get_buffer_size())

    output_dir = prepare_output_dir(args=None, user_specified_dir="./results")
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
                    samples["rew"], np.array(samples["done"], dtype=np.float64),
                    samples["weights"])
                writer.flush()
            global_rb.update_priorities(samples["indexes"], np.abs(td_errors) + 1e-6)
            lock.release()

            # Put updated weights to queue
            if trained_steps.value % update_freq == 0:
                queues[0].put(get_weights_fn(policy))
                with tf.contrib.summary.always_record_summaries():
                    fps = update_freq / (time.time() - start_time)
                    tf.contrib.summary.scalar(name="FPS", tensor=fps, family="loss")
                    print("Update weights for explorer. {0:.2f} FPS for GRAD. Learned {1:.2f} steps".format(
                        fps, trained_steps.value))
                start_time = time.time()
            if trained_steps.value % evaluation_freq == 0:
                queues[1].put(get_weights_fn(policy))


        if trained_steps.value >= n_training:
            is_training_done.set()


def evaluator(is_training_done, env, policy_fn, set_weights_fn, queue,
              n_evaluation=10, episode_max_steps=1000, show_test_progress=False):
    policy = policy_fn(env, "Learner")

    while not is_training_done.is_set():
        if queue.empty():
            continue
        else:
            set_weights_fn(policy, queue.get())
            avg_test_return = 0.
            for i in range(n_evaluation):
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
            print("Evaluation: {}".format(avg_test_return / n_evaluation))


def apex_argument(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--max-batch', type=int, default=1e6,
                        help='Number of times to apply batch update')
    parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                        help='Maximum steps in an episode')
    parser.add_argument('--param-update-freq', type=int, default=1e2,
                        help='Frequency to update parameter')
    parser.add_argument('--test-freq', type=int, default=1e3,
                        help='Frequency to evaluate policy')
    parser.add_argument('--n-env', type=int, default=64,
                        help='Number of environments')
    parser.add_argument('--n-thread', type=int, default=4,
                        help='Number of thread pool')
    parser.add_argument('--replay-buffer-size', type=int, default=1e6,
                        help='Size of replay buffer')
    parser.add_argument('--local-buffer-size', type=int, default=1e4,
                        help='Size of local replay buffer for explorer')
    parser.add_argument('--gpu', type=int, default=0)
    return parser


def prepare_experiment(env, args):
    # Manager to share PER between a learner and explorers
    SyncManager.register('PrioritizedReplayBuffer',
                         PrioritizedReplayBuffer)
    manager = SyncManager()
    manager.start()

    global_rb = manager.PrioritizedReplayBuffer(
        obs_shape=env.observation_space.shape,
        act_dim=get_act_dim(env),
        size=args.replay_buffer_size)

    # queues to share network parameters between a learner and explorers
    queues = [Queue(), Queue()]

    # Event object to share training status. if event is set True, all exolorers stop sampling transitions
    is_training_done = Event()

    # Lock
    lock = manager.Lock()

    # Shared memory objects to count number of samples and applied gradients
    trained_steps = Value('i', 0)

    return global_rb, queues, is_training_done, lock, trained_steps
