import time
import numpy as np
import tensorflow as tf
import argparse

import multiprocessing
from multiprocessing import Process, Queue, Value, Event, Lock
from multiprocessing.managers import SyncManager

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from tf2rl.misc.prepare_output_dir import prepare_output_dir

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


def explorer(global_rb, queue, trained_steps, n_transition,
             is_training_done, lock, env, policy_fn,
             set_weights_fn, noise,
             buffer_size=1024, max_transition=None,
             episode_max_steps=1000):
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
        n_transition:
            Number of collected transitions.
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
    policy = policy_fn(env, "Explorer", global_rb.get_buffer_size(), sigma=noise)
    local_rb = ReplayBuffer(obs_shape=env.observation_space.shape,
                            act_dim=env.action_space.low.size,
                            size=buffer_size)

    s = env.reset()
    episode_steps = 0
    total_reward = 0.
    total_rewards = []
    start = time.time()
    sample_at_start = 0

    while not is_training_done.is_set():
        n_transition.value += 1
        episode_steps += 1
        a = policy.get_action(s)
        s_, r, done, _ = env.step(a)
        done_flag = done
        if episode_steps == env._max_episode_steps:
            done_flag = False
        total_reward += r
        local_rb.add(s, a, r, s_, done_flag)

        s = s_
        if done or episode_steps == episode_max_steps:
            s = env.reset()
            total_rewards.append(total_reward)
            total_reward = 0
            episode_steps = 0

            # Periodically copy weights of explorer
            if not queue.empty():
                set_weights_fn(policy, queue.get())

        # Add collected experiences to global replay buffer
        if local_rb.get_stored_size() == buffer_size:
            samples = local_rb.sample(local_rb.get_stored_size())
            td_errors = policy.compute_td_error(
                samples["obs"], samples["act"], samples["next_obs"],
                samples["rew"], np.array(samples["done"], dtype=np.float64))
            msg = "Grad: {0: 6d}\t".format(trained_steps.value)
            msg += "Samples: {0: 7d}\t".format(n_transition.value)
            msg += "TDErr: {0:.5f}\t".format(np.average(np.abs(td_errors).flatten()))
            msg += "AveEpiRew: {0:.3f}\t".format(sum(total_rewards) / len(total_rewards))
            msg += "FPS: {0:.2f}".format((n_transition.value - sample_at_start) / (time.time() - start))
            print(msg)
            total_rewards = []
            lock.acquire()
            global_rb.add(
                samples["obs"], samples["next_obs"], samples["rew"], samples["next_obs"], samples["done"],
                priorities=np.abs(td_errors)+1e-6)
            lock.release()
            local_rb.clear()
            start = time.time()
            sample_at_start = n_transition.value

        if max_transition is not None and n_transition.value >= max_transition:
            is_training_done.set()


def learner(global_rb, trained_steps, is_training_done,
            lock, env, policy_fn, get_weights_fn,
            n_training, update_freq, *queues):
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
                weights = get_weights_fn(policy)
                for queue in queues:
                    queue.put(weights)
                with tf.contrib.summary.always_record_summaries():
                    fps = update_freq / (time.time() - start_time)
                    tf.contrib.summary.scalar(name="FPS", tensor=fps, family="loss")
                    print("Update weights for explorer. {0:.2f} FPS for GRAD. Learned {1:.2f} steps".format(
                        fps, trained_steps.value))
                start_time = time.time()

        if trained_steps.value >= n_training:
            is_training_done.set()


def apex_argument(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--max-batch', type=int, default=1e6,
                        help='number of times to apply batch update')
    parser.add_argument('--param-update-freq', type=int, default=1e2,
                        help='frequency to update parameter')
    parser.add_argument('--n-explorer', type=int, default=None,
                        help='number of explorers to distribute. if None, use maximum number')
    parser.add_argument('--replay-buffer-size', type=int, default=1e6,
                        help='size of replay buffer')
    parser.add_argument('--local-buffer-size', type=int, default=1e4,
                        help='size of local replay buffer for explorer')
    parser.add_argument('--gpu', type=int, default=0)
    return parser


def prepare_experiment(n_explorer, env, args):
    # Manager to share PER between a learner and explorers
    SyncManager.register('PrioritizedReplayBuffer',
                         PrioritizedReplayBuffer)
    manager = SyncManager()
    manager.start()
    global_rb = manager.PrioritizedReplayBuffer(
        obs_shape=env.observation_space.shape,
        act_dim=env.action_space.low.size,
        size=args.replay_buffer_size)

    # queues to share network parameters between a learner and explorers
    queues = [Queue() for _ in range(n_explorer)]

    # Event object to share training status. if event is set True, all exolorers stop sampling transitions
    is_training_done = Event()

    # Lock
    lock = manager.Lock()

    # Shared memory objects to count number of samples and applied gradients
    trained_steps = Value('i', 0)
    n_transition = Value('i', 0)

    return global_rb, queues, is_training_done, lock, trained_steps, n_transition
