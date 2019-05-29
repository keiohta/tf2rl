import os
import time
import numpy as np
import logging
import argparse
import tensorflow as tf

from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.experiments.utils import save_path


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class Trainer:
    def __init__(
            self,
            policy,
            env,
            args,
            test_env=None):
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env
        self._set_from_args(args)

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir="./results", suffix=args.dir_suffix)
        logging.basicConfig(level=logging.getLevelName(args.logging_level))
        self.logger = logging.getLogger(__name__)

        # Save and restore model
        checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(
            checkpoint, directory=self._output_dir, max_to_keep=5)
        if args.model_dir is not None:
            assert os.path.isdir(args.model_dir)
            path_ckpt = tf.train.latest_checkpoint(args.model_dir)
            checkpoint.restore(path_ckpt)
            self.logger.info("Restored {}".format(path_ckpt))

        # prepare TensorBoard output
        self.writer = tf.contrib.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()
        tf.contrib.summary.initialize()

    def __call__(self):
        tf_total_steps = tf.train.create_global_step()
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.time()
        n_episode = 0

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        obs = self._env.reset()

        with tf.contrib.summary.record_summaries_every_n_global_steps(1000):
            while total_steps < self._max_steps:
                if total_steps < self._policy.n_warmup:
                    action = self._env.action_space.sample()
                else:
                    action = self._policy.get_action(obs)

                next_obs, reward, done, _ = self._env.step(action)
                if self._show_progress:
                    self._env.render()
                episode_steps += 1
                episode_return += reward
                tf_total_steps.assign_add(1)
                total_steps += 1

                done_flag = done
                if hasattr(self._env, "_max_episode_steps") and \
                        episode_steps == self._env._max_episode_steps:
                    done_flag = False
                replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag)
                obs = next_obs

                if done or episode_steps == self._episode_max_steps:
                    obs = self._env.reset()

                    n_episode += 1
                    fps = episode_steps / (time.time() - episode_start_time)
                    self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                        n_episode, int(total_steps), episode_steps, episode_return, fps))

                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.time()

                if int(total_steps) >= self._policy.n_warmup and int(total_steps) % self._policy.update_interval == 0:
                    samples = replay_buffer.sample(self._policy.batch_size)
                    td_error = self._policy.train(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32),
                        None if not self._use_prioritized_rb else samples["weights"])
                    if self._use_prioritized_rb:
                        replay_buffer.update_priorities(samples["indexes"], np.abs(td_error) + 1e-6)
                    if int(total_steps) % self._test_interval == 0:
                        with tf.contrib.summary.always_record_summaries():
                            avg_test_return = self.evaluate_policy(int(total_steps))
                            self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                                int(total_steps), avg_test_return, self._test_episodes))
                            tf.contrib.summary.scalar(name="AverageTestReturn", tensor=avg_test_return, family="loss")
                            tf.contrib.summary.scalar(name="FPS", tensor=fps, family="loss")

                        self.writer.flush()

                if int(total_steps) % self._model_save_interval == 0:
                    self.checkpoint_manager.save()

            tf.contrib.summary.flush()

    def evaluate_policy(self, total_steps):
        avg_test_return = 0.
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        for i in range(self._test_episodes):
            episode_return = 0.
            obs = self._test_env.reset()
            done = False
            for _ in range(self._episode_max_steps):
                action = self._policy.get_action(obs, test=True)
                next_obs, reward, done, _ = self._test_env.step(action)
                if self._save_test_path:
                    replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)
                if self._show_test_progress:
                    self._test_env.render()
                episode_return += reward
                obs = next_obs
                if done:
                    break
            if self._save_test_path:
                filename = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}.pkl".format(
                    total_steps, i, episode_return)
                save_path(replay_buffer.sample(self._episode_max_steps),
                          os.path.join(self._output_dir, filename))
                replay_buffer.clear()
            avg_test_return += episode_return

        return avg_test_return / self._test_episodes

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = args.episode_max_steps \
            if args.episode_max_steps is not None \
            else args.max_steps
        self._show_progress = args.show_progress
        self._model_save_interval = args.save_model_interval
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--gpu', type=int, default=0,
                            help='GPU id')
        parser.add_argument('--save-model-interval', type=int, default=int(1e4),
                            help='Interval to save model')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        # test settings
        parser.add_argument('--test-interval', type=int, default=int(1e4),
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=5,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation')
        # replay buffer
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')
        return parser
