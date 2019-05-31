import time
import numpy as np
import tensorflow as tf

from tf2rl.experiments.trainer import Trainer
from tf2rl.misc.get_replay_buffer import get_replay_buffer


class OnPolicyTrainer(Trainer):
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
                for _ in range(self._policy.horizon):
                    action, log_pi = self._policy.get_action(obs)
                    next_obs, reward, done, _ = self._env.step(action)
                    if self._show_progress:
                        self._env.render()
                    episode_steps += 1
                    episode_return += reward
                    total_steps += 1

                    done_flag = done
                    if hasattr(self._env, "_max_episode_steps") and \
                            episode_steps == self._env._max_episode_steps:
                        done_flag = False
                    replay_buffer.add(obs=obs, act=action, next_obs=next_obs,
                                      rew=reward, done=done_flag, log_pi=log_pi)
                    obs = next_obs

                    if done or episode_steps == self._episode_max_steps:
                        obs = self._env.reset()
                        tf_total_steps.assign_add(episode_steps)
                        n_episode += 1
                        fps = episode_steps / (time.time() - episode_start_time)
                        self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                            n_episode, int(total_steps), episode_steps, episode_return, fps))

                        episode_steps = 0
                        episode_return = 0
                        episode_start_time = time.time()

                samples = replay_buffer.sample(self._policy.batch_size)
                self._policy.train(
                    samples["obs"], samples["act"], samples["next_obs"],
                    samples["rew"], np.array(samples["done"], dtype=np.float32),
                    None if not self._use_prioritized_rb else samples["weights"])
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

