"""
multi環境を同時に扱うための環境
"""

import time
import argparse
import logging
import threading

import numpy as np
import tensorflow as tf
import gym
import gin
from tf2rl.algos.sac import SAC as SAC_tf2rl
from tf2rl.algos.ddpg import DDPG as DDPG_tf2rl

import gatling.util.gin_tf_external
import gatling.util.misc as misc
from gatling.util.setting import make_output_dir
from gatling.fast_env.shmem_env import ShmemEnv
from gatling.policy import DDPG, SAC
from gatling.fast_env.mujoco_env_wrapper import MuJoCoEnvWrapper
from tf_replay_buffer import ReplayBuffer

misc.set_gpu_device_growth()


def make_policy(env, name, tf2rl=False):
    dim_state, dim_action = env.observation_space.shape[0], env.action_space.shape[0]
    max_action = env.action_space.high

    with tf.device("/gpu:0"):
        if name == "DDPG":
            if tf2rl:
                policy = DDPG_tf2rl(state_shape=(dim_state,), action_dim=dim_action,
                                    max_action=max_action[0], max_grad=1.)
                saved_policy = DDPG_tf2rl(state_shape=(dim_state,), action_dim=dim_action,
                                          max_action=max_action[0], max_grad=1.)
            else:
                policy = DDPG.DDPG(dim_state, dim_action, max_action, training=True)
                saved_policy = DDPG.DDPG(dim_state, dim_action, max_action, training=False)
        elif name == "SAC":
            if tf2rl:
                policy = SAC_tf2rl(state_shape=(dim_state,), action_dim=dim_action, max_action=max_action[0])
                saved_policy = SAC_tf2rl(state_shape=(dim_state,), action_dim=dim_action, max_action=max_action[0])
            else:
                policy = SAC.SAC(dim_state, dim_action, max_action, training=True)
                saved_policy = SAC.SAC(dim_state, dim_action, max_action, training=False)
        else:
            raise ValueError("invalid policy")

    return policy, saved_policy


@gin.configurable(blacklist=["dim_state", "dim_action"])
def make_replay_buffer(dim_state, dim_action, alpha=0.6, beta=0.4, eps=1e-3, size=1000000):
    with tf.name_scope("ReplayBuffer"):
        prioritized_replay_alpha = alpha
        prioritized_replay_beta0 = beta
        prioritized_replay_eps = eps
        replay_buffer_size = size

        # states, actions, next_states, rewards, dones
        buffer_shapes = [tf.TensorShape([dim_state]), tf.TensorShape([dim_action]), tf.TensorShape([dim_state]),
                         tf.TensorShape([]), tf.TensorShape([])]
        buffer_dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, bool]

        replay_buffer = ReplayBuffer(replay_buffer_size,
                                     shapes=buffer_shapes,
                                     dtypes=buffer_dtypes,
                                     alpha=prioritized_replay_alpha,
                                     beta=prioritized_replay_beta0,
                                     priority_eps=prioritized_replay_eps)

    return replay_buffer


def collect_transitions(dir_log, vec_env, policy, replay_buffer, cur_steps, random_transitions=1000):
    """transitionsを収集する

    :param ShmemEnv vec_env:
    :param policy:
    :param replay_buffer:
    :param cur_steps:
    :return:
    """
    num_envs = vec_env.num_envs
    num_collects = 0
    log_freq = 100000
    next_log_steps = log_freq
    perf_counter = misc.PerfCounter(name="Collect")
    perf_counter.start(num_collects)

    writer = tf.summary.create_file_writer(dir_log)
    writer.set_as_default()

    while True:
        states = vec_env.observation()

        if num_collects < random_transitions:
            actions = np.random.uniform(vec_env.action_space.low, vec_env.action_space.high,
                                        vec_env.action_shape).astype(np.float32)
        else:
            with tf.device("/gpu:0"):
                # TODO 複数のノイズレベルに対応すべき
                actions = policy.get_action(states)

        next_states, rewards, dones, reach_limits = vec_env.step(actions)
        dones = np.logical_xor(dones, reach_limits)

        components = [states, actions, next_states, rewards, dones]

        with tf.device("/gpu:0"):
            td_error = policy.compute_td_error(states, actions, next_states, rewards, dones)
            priority = tf.abs(td_error)

        replay_buffer.enqueue_many(components, priority)
        num_collects += num_envs

        if num_collects > next_log_steps:
            next_log_steps += log_freq
            throughput = perf_counter.stop(num_collects)
            tf.summary.scalar("perf/collect_transitions", num_collects, step=cur_steps)
            tf.summary.scalar(name="perf/collect_throughput", data=throughput, step=cur_steps)

            perf_counter.start(num_collects)


@tf.function
def learn(policy, replay_buffer, batch_size, discount):
    """1ステップ分学習する. メインスレッドから実行される

    :return:
    """
    idx, weights, components = replay_buffer.sample(batch_size=batch_size)
    sample_states, sample_actions, sample_nexts, sample_rewards, sample_dones = components

    with tf.device("/gpu:0"):
        policy.train(sample_states, sample_actions, sample_nexts, sample_rewards, sample_dones, weights)
        td_error = policy.compute_td_error(sample_states, sample_actions, sample_nexts, sample_rewards, sample_dones)

        tf.summary.histogram("debug/weights", data=weights)

        new_priorities = tf.abs(td_error)
        replay_buffer.assign_with_eps(idx, new_priorities)


def main():
    logger = gatling.util.setting.set_root_logger()

    parser = argparse.ArgumentParser()
    # RoboschoolHalfCheetah-v1, Pendulum-v0
    parser.add_argument("--env", default="RoboschoolAnt-v1")
    # parser.add_argument("--env", default="Pendulum-v0")
    parser.add_argument("--policy", default="SAC")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--steps", default=3000000, type=int)
    parser.add_argument("--gin", default=None)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--force", default=False, action="store_true", help="remove existed directory")
    parser.add_argument("--tf2rl", action="store_true")
    args = parser.parse_args()

    env_name, policy_name = args.env, args.policy
    max_steps = args.steps
    seed = args.seed
    discount_rate = 0.99
    exp_name = args.env.split("-")[0]
    dir_log, dir_log_collect, dir_parameter, dir_saved = make_output_dir(
        exp_name=exp_name, seed=seed, force=args.force)

    writer = tf.summary.create_file_writer(dir_log)
    writer.set_as_default()

    if args.gin is not None:
        gin.parse_config_file(args.gin)

    summary_freq = 5000
    eval_freq = 2000
    save_freq = 5000
    num_processes = 4
    random_transitions = 10000
    envs_per_process = 128
    batch_size = 250

    def make_env():
        env = gym.make(env_name)
        if env.env.__class__.__bases__[0] == gym.envs.mujoco.mujoco_env.MujocoEnv:
            return MuJoCoEnvWrapper(env)
        else:
            env

    eval_env = make_env()
    vec_env = ShmemEnv(make_env, num_processes, envs_per_process, eval_env.observation_space, eval_env.action_space)

    dim_state, dim_action = eval_env.observation_space.shape[0], eval_env.action_space.shape[0]
    replay_buffer = make_replay_buffer(dim_state, dim_action)
    policy, saved_policy = make_policy(env=eval_env, name=policy_name, tf2rl=args.tf2rl)

    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=dir_parameter,
                                                    max_to_keep=1)

    saved_checkpoint = tf.train.Checkpoint(policy=saved_policy)
    steps = tf.Variable(0, dtype=np.int64)
    tf.summary.experimental.set_step(steps)

    should_summary = lambda: tf.equal(steps % summary_freq, 0)
    logger.info("start training")

    collect_thread = threading.Thread(target=collect_transitions,
                                      args=(dir_log_collect, vec_env, policy, replay_buffer, steps, random_transitions))
    collect_thread.start()

    perf_counter = misc.PerfCounter(name="Train")
    perf_counter.start(0)
    # sampleする前に最低限のtransitionsが貯まるまで待つ
    time.sleep(3)

    while steps <= max_steps:
        steps.assign_add(1)
        cur_step = int(steps.numpy())

        with tf.summary.record_if(should_summary):
            learn(policy, replay_buffer, batch_size, discount=discount_rate)

        if cur_step % eval_freq == 0:
            total_transitions = cur_step * batch_size
            tf.summary.scalar("perf/train_transitions", total_transitions, step=steps)
            throughput = perf_counter.stop(total_transitions)
            tf.summary.scalar(name="perf/train_throughput", data=throughput)
            evaluate_policy(eval_env, policy, trained_transitions=cur_step * batch_size)
            perf_counter.start(total_transitions)

        if cur_step % save_freq == 0:
            checkpoint_manager.save(cur_step)
            saved_checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            # sig = saved_policy.get_saved_signature()
            # tf.saved_model.save(saved_policy, dir_saved, signatures=sig)


@gin.configurable(whitelist=["eval_episodes"])
def evaluate_policy(env, policy, trained_transitions, eval_episodes=10):
    logger = logging.getLogger("gatling")
    episode_returns = []
    episode_length = []

    for _ in range(eval_episodes):
        state = env.reset()
        cur_length = 0
        cur_return = 0

        done = False
        while not done:
            with tf.device("/gpu:0"):
                action = policy.get_action(np.array(state, dtype=np.float32))
            state, reward, done, _ = env.step(action)
            cur_return += reward
            cur_length += 1

        episode_returns.append(cur_return)
        episode_length.append(cur_length)

    mean_return = np.mean(episode_returns)
    mean_length = np.mean(episode_length)

    logger.info(f"Trained {trained_transitions} transisions Eval Average Reward {mean_return}")
    tf.summary.scalar(name="loss/evaluate_return", data=mean_return)
    tf.summary.scalar(name="loss/evaluate_steps", data=mean_length)


if __name__ == "__main__":
    main()
