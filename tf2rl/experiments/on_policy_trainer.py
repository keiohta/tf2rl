import os
import time

import numpy as np
import tensorflow as tf

from cpprb.experimental import ReplayBuffer

from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.envs.utils import is_discrete


class OnPolicyTrainer(Trainer):
    def __call__(self):
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.time()
        n_episode = 0
        test_step_threshold = self._test_interval

        # TODO: clean codes
        self.replay_buffer = get_replay_buffer(
            self._policy, self._env)
        kwargs_local_buf = get_default_rb_dict(
            size=self._episode_max_steps, env=self._env)
        kwargs_local_buf["env_dict"]["logp"] = {}
        kwargs_local_buf["env_dict"]["val"] = {}
        if is_discrete(self._env.action_space):
            kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
        self.local_buffer = ReplayBuffer(**kwargs_local_buf)

        obs = self._env.reset()
        while total_steps < self._max_steps:
            for _ in range(self._policy.horizon):
                action, log_pi, val = self._policy.get_action_and_val(obs)
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
                self.local_buffer.add(
                    obs=obs, act=action, next_obs=next_obs,
                    rew=reward, done=done_flag, logp=log_pi, val=val)
                obs = next_obs

                if done or episode_steps == self._episode_max_steps:
                    self.finish_horizon()
                    obs = self._env.reset()
                    n_episode += 1
                    fps = episode_steps / (time.time() - episode_start_time)
                    self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                        n_episode, int(total_steps), episode_steps, episode_return, fps))

                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.time()

            self.finish_horizon(last_val=val)
            tf.summary.experimental.set_step(total_steps)
            samples = self.replay_buffer.sample(self._policy.horizon)
            # Normalize advantages
            if self._policy.normalize_adv:
                adv = (samples["adv"] - np.mean(samples["adv"])
                       ) / np.std(samples["adv"])
            else:
                adv = samples["adv"]
            for _ in range(1):
                self._policy.train_actor(
                    samples["obs"],
                    samples["act"],
                    adv,
                    samples["logp"])
            # Train Critic
            for _ in range(5):
                self._policy.train_critic(
                    samples["obs"],
                    samples["ret"])
            if total_steps > test_step_threshold:
                test_step_threshold += self._test_interval
                avg_test_return = self.evaluate_policy(total_steps)
                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))
                tf.summary.scalar(
                    name="Common/average_test_return", data=avg_test_return)
                tf.summary.scalar(name="Common/fps", data=fps)

                self.writer.flush()

            if total_steps % self._model_save_interval == 0:
                self.checkpoint_manager.save()

        tf.summary.flush()

    def finish_horizon(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        samples = self.local_buffer._encode_sample(
            np.arange(self.local_buffer.get_stored_size()))
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]
        if self._policy.enable_gae:
            advs = discount_cumsum(
                deltas, self._policy.discount * self._policy.lam)
        else:
            advs = deltas

        # Rewards-to-go, to be targets for the value function
        rets = discount_cumsum(rews, self._policy.discount)[:-1]
        self.replay_buffer.add(
            obs=samples["obs"], act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
        self.local_buffer.clear()

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
                action, _ = self._policy.get_action(obs, test=True)
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
                tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
                tf.uint8)
            tf.summary.image('train/input_img', images,)
        return avg_test_return / self._test_episodes
