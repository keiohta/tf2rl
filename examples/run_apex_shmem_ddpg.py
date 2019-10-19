"""
multi環境を同時に扱うための環境
"""

import argparse
import logging

import gatling.util.gin_tf_external
import gatling.util.misc as misc
import gin
import gym
import numpy as np
import tensorflow as tf

from gatling.policy import DDPG, SAC
from gatling.replay_buffer import segment_tree
from gatling.util.setting import make_output_dir
from tf2rl.envs.share_mem_env import ShmemEnv
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.misc.prepare_output_dir import prepare_output_dir


def make_policy(env, name):
    dim_state, dim_action = env.observation_space.shape[0], env.action_space.shape[0]
    max_action = env.action_space.high

    with tf.device("/gpu:0"):
        if name == "DDPG":
            policy = DDPG.DDPG(dim_state, dim_action, max_action, training=True)
            saved_policy = DDPG.DDPG(dim_state, dim_action, max_action, training=False)
        elif name == "SAC":
            policy = SAC.SAC(dim_state, dim_action, max_action, training=True)
            saved_policy = SAC.SAC(dim_state, dim_action, max_action, training=False)
        else:
            raise ValueError("invalid policy")

    return policy, saved_policy


def make_replay_buffer(dim_state, dim_action):
    # buffer = replay.ReplayBuffer(dim_state=dim_state, dim_action=dim_action, capacity=1000000)

    with tf.name_scope("ReplayBuffer"):
        prioritized_replay_alpha = 0.6
        prioritized_replay_beta0 = 0.4
        prioritized_replay_eps = 1e-6
        replay_buffer_size = 1000000

        # states, actions, next_states, rewards, dones
        buffer_shapes = [tf.TensorShape([dim_state]), tf.TensorShape([dim_action]), tf.TensorShape([dim_state]),
                         tf.TensorShape([]), tf.TensorShape([])]
        buffer_dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, bool]

        replay_buffer = segment_tree.ReplayBuffer(replay_buffer_size,
                                                  shapes=buffer_shapes,
                                                  dtypes=buffer_dtypes,
                                                  alpha=prioritized_replay_alpha,
                                                  beta=prioritized_replay_beta0,
                                                  priority_eps=prioritized_replay_eps)

    return replay_buffer


def main():
    logger = initialize_logger()

    parser = argparse.ArgumentParser()
    # RoboschoolHalfCheetah-v1, Pendulum-v0
    # parser.add_argument("--env", default="RoboschoolHalfCheetah-v1")
    parser.add_argument("--env", default="Pendulum-v0")
    parser.add_argument("--policy", default="SAC")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--steps", default=1000000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--force", default=False, action="store_true", help="remove existed directory")
    args = parser.parse_args()

    env_name, policy_name = args.env, args.policy
    batch_size, max_steps = args.batch_size, args.steps
    seed = args.seed
    exp_name = args.env.split("-")[0]
    dir_log, _, dir_parameter, dir_saved = make_output_dir(exp_name=exp_name, seed=seed, force=args.force)

    writer = tf.summary.create_file_writer(dir_log)
    writer.set_as_default()

    summary_freq = 1000
    eval_freq = 5000
    save_freq = 5000
    num_processes = 4
    random_transitions = 10000
    envs_per_process = 25
    num_envs = num_processes * envs_per_process

    def make_env():
        return gym.make(env_name)

    eval_env = make_env()
    vec_env = ShmemEnv(make_env, num_processes, envs_per_process,
                       eval_env.observation_space, eval_env.action_space)

    dim_state, dim_action = eval_env.observation_space.shape[0], eval_env.action_space.shape[0]
    action_shape = (num_envs,) + eval_env.action_space.shape
    replay_buffer = make_replay_buffer(dim_state, dim_action)
    policy, saved_policy = make_policy(env=eval_env, name=policy_name)

    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=dir_parameter, max_to_keep=1)

    saved_checkpoint = tf.train.Checkpoint(policy=saved_policy)
    total_transitions = tf.Variable(0, dtype=np.int64)
    tf.summary.experimental.set_step(total_transitions)

    should_summary = lambda: tf.equal(total_transitions % summary_freq, 0)
    logger.info("start training")

    while total_transitions <= max_steps:
        states = vec_env.observation()

        if total_transitions < random_transitions:
            actions = np.random.uniform(eval_env.action_space.low, eval_env.action_space.high, action_shape).astype(
                np.float32)
        else:
            with tf.device("/gpu:0"):
                actions = policy.select_noisy_actions(states).numpy()

        next_states, rewards, dones, reach_limits = vec_env.step(actions)
        dones = np.logical_xor(dones, reach_limits)

        total_transitions.assign_add(num_envs)
        cur_transitions = int(total_transitions.numpy())
        components = [states, actions, next_states, rewards, dones]

        with tf.device("/gpu:0"):
            td_error = policy.td_error(states, actions, next_states, rewards, dones)
            priority = tf.abs(td_error)

        replay_buffer.enqueue_many(components, priority)

        if cur_transitions < random_transitions:
            continue

        idx, weights, components = replay_buffer.sample(batch_size=batch_size)
        sample_states, sample_actions, sample_nexts, sample_rewards, sample_dones = components

        with tf.summary.record_if(should_summary):
            with tf.device("/gpu:0"):
                policy.train(sample_states, sample_actions, sample_nexts, sample_rewards, sample_dones)
                td_error = policy.td_error(sample_states, sample_actions, sample_nexts, sample_rewards, sample_dones)

                new_priorities = tf.abs(td_error)
                replay_buffer.assign_with_eps(idx, new_priorities)

        if cur_transitions % eval_freq == 0:
            evaluate_policy(eval_env, policy)

        if cur_transitions % save_freq == 0:
            checkpoint_manager.save(cur_transitions)
            saved_checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            sig = saved_policy.get_saved_signature()
            tf.saved_model.save(saved_policy, dir_saved, signatures=sig)


@gin.configurable(whitelist=["eval_episodes"])
def evaluate_policy(env, policy, eval_episodes=10):
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
                action = policy.numpy_select_action(np.array(state, dtype=np.float32))
            state, reward, done, _ = env.step(action)
            cur_return += reward
            cur_length += 1

        episode_returns.append(cur_return)
        episode_length.append(cur_length)

    mean_return = np.mean(episode_returns)
    mean_length = np.mean(episode_length)

    logger.info("Eval Average Reward {}".format(mean_return))
    tf.summary.scalar(name="loss/evaluate_return", data=mean_return)
    tf.summary.scalar(name="loss/evaluate_steps", data=mean_length)


@gin.configurable(whitelist=["num_transitions"])
def collect_random_transitions(env, replay_buffer, num_transitions=10000):
    """ collects transitions with a random action policy

    This procedure follows TD3 setting.

    :return:
    """
    episode_return = 0
    episode_timesteps = 0

    state = env.reset()

    # collect transitions by random action
    for i in range(num_transitions):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        episode_return += reward
        episode_timesteps += 1

        done_flag = done
        # noinspection PyProtectedMember
        if episode_timesteps == env._max_episode_steps:
            done_flag = False

        replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done_flag)
        state = next_state

        if done:
            state = env.reset()
            episode_timesteps = 0
            episode_return = 0

    return num_transitions


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )

    main()
