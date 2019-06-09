import time
import numpy as np
import tensorflow as tf

from tf2rl.experiments.trainer import Trainer
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.experiments.utils import save_path, frames_to_gif


class OnPolicyTrainer(Trainer):
    def __call__(self):
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.time()
        n_episode = 0
        test_step_threshold = self._test_interval

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        obs = self._env.reset()
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
                    n_episode += 1
                    fps = episode_steps / (time.time() - episode_start_time)
                    self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                        n_episode, int(total_steps), episode_steps, episode_return, fps))

                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.time()

            tf.summary.experimental.set_step(total_steps)
            idxes = np.arange(self._policy.horizon)
            samples = replay_buffer.sample(self._policy.horizon)
            np.random.shuffle(idxes)
            for i in range(int(self._policy.horizon / self._policy.batch_size / 2)):
                idx = i * 2 * self._policy.batch_size
                # Train critic
                self._policy.train_critic(
                    samples["obs"][idx:idx+self._policy.batch_size],
                    samples["act"][idx:idx+self._policy.batch_size],
                    samples["next_obs"][idx:idx+self._policy.batch_size],
                    samples["rew"][idx:idx+self._policy.batch_size],
                    samples["done"][idx:idx+self._policy.batch_size])
                # Train actor
                idx += self._policy.batch_size
                self._policy.train_actor(
                    samples["obs"][idx:idx+self._policy.batch_size],
                    samples["act"][idx:idx+self._policy.batch_size],
                    samples["next_obs"][idx:idx+self._policy.batch_size],
                    samples["rew"][idx:idx+self._policy.batch_size],
                    samples["done"][idx:idx+self._policy.batch_size],
                    samples["log_pi"][idx:idx+self._policy.batch_size])
            if total_steps > test_step_threshold == 0:
                test_step_threshold += self._test_interval
                avg_test_return = self.evaluate_policy(total_steps)
                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))
                tf.summary.scalar(name="AverageTestReturn", data=avg_test_return, description="loss")
                tf.summary.scalar(name="FPS", data=fps, description="loss")

                self.writer.flush()

            if total_steps % self._model_save_interval == 0:
                self.checkpoint_manager.save()

        tf.summary.flush()

    def evaluate_policy(self, total_steps):
        avg_test_return = 0.
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            obs = self._test_env.reset()
            done = False
            for _ in range(self._episode_max_steps):
                action, log_pi = self._policy.get_action(obs, test=True)
                next_obs, reward, done, _ = self._test_env.step(action)
                if self._save_test_path:
                    replay_buffer.add(
                        obs=obs, act=action, next_obs=next_obs,
                        rew=reward, done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))
                elif self._show_test_progress:
                    self._test_env.render()
                episode_return += reward
                obs = next_obs
                if done:
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                total_steps, i, episode_return)
            if self._save_test_path:
                save_path(replay_buffer.sample(self._episode_max_steps),
                          os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return
        if self._show_test_images:
            images = tf.cast(
                tf.expand_dims(np.array(obs).transpose(2,0,1), axis=3),
                tf.uint8)
            tf.summary.image('train/input_img', images,)
        return avg_test_return / self._test_episodes
