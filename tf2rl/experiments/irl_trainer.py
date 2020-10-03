import time

import numpy as np
import tensorflow as tf

from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.experiments.trainer import Trainer


class IRLTrainer(Trainer):
    def __init__(
            self,
            policy,
            env,
            args,
            irl,
            expert_obs,
            expert_next_obs,
            expert_act,
            test_env=None):
        self._irl = irl
        args.dir_suffix = self._irl.policy_name + args.dir_suffix
        super().__init__(policy, env, args, test_env)
        # TODO: Add assertion to check dimention of expert demos and current policy, env is the same
        self._expert_obs = expert_obs
        self._expert_next_obs = expert_next_obs
        self._expert_act = expert_act
        # Minus one to get next obs
        self._random_range = range(expert_obs.shape[0])

    def __call__(self):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        obs = self._env.reset()

        while total_steps < self._max_steps:
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
                total_steps += 1
                tf.summary.experimental.set_step(total_steps)

                done_flag = done
                if (hasattr(self._env, "_max_episode_steps") and
                    episode_steps == self._env._max_episode_steps):
                    done_flag = False
                replay_buffer.add(obs=obs, act=action,
                                  next_obs=next_obs, rew=reward, done=done_flag)
                obs = next_obs

                if done or episode_steps == self._episode_max_steps:
                    replay_buffer.on_episode_end()
                    obs = self._env.reset()

                    n_episode += 1
                    fps = episode_steps / (time.perf_counter() - episode_start_time)
                    self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                        n_episode, int(total_steps), episode_steps, episode_return, fps))
                    tf.summary.scalar(
                        name="Common/training_return", data=episode_return)

                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.perf_counter()

                if total_steps < self._policy.n_warmup:
                    continue

                if total_steps % self._policy.update_interval == 0:
                    samples = replay_buffer.sample(self._policy.batch_size)
                    # Train policy
                    rew = self._irl.inference(samples["obs"], samples["act"], samples["next_obs"])
                    with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                        self._policy.train(
                            samples["obs"], samples["act"], samples["next_obs"],
                            rew, samples["done"],
                            None if not self._use_prioritized_rb else samples["weights"])
                        if self._use_prioritized_rb:
                            td_error = self._policy.compute_td_error(
                                samples["obs"], samples["act"], samples["next_obs"],
                                rew, samples["done"])
                            replay_buffer.update_priorities(
                                samples["indexes"], np.abs(td_error) + 1e-6)

                        # Train IRL
                        for _ in range(self._irl.n_training):
                            samples = replay_buffer.sample(self._irl.batch_size)
                            # Do not allow duplication!!!
                            indices = np.random.choice(
                                self._random_range, self._irl.batch_size, replace=False)
                            self._irl.train(
                                agent_states=samples["obs"],
                                agent_acts=samples["act"],
                                agent_next_states=samples["next_obs"],
                                expert_states=self._expert_obs[indices],
                                expert_acts=self._expert_act[indices],
                                expert_next_states=self._expert_next_obs[indices])

                if total_steps % self._test_interval == 0:
                    avg_test_return = self.evaluate_policy(total_steps)
                    self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                        total_steps, avg_test_return, self._test_episodes))
                    tf.summary.scalar(
                        name="Common/average_test_return", data=avg_test_return)
                    tf.summary.scalar(
                        name="Common/fps", data=fps)
                    self.writer.flush()

                if total_steps % self._save_model_interval == 0:
                    self.checkpoint_manager.save()

            tf.summary.flush()

    @staticmethod
    def get_argument(parser=None):
        parser = Trainer.get_argument(parser)
        parser.add_argument('--expert-path-dir', default=None,
                            help='Path to directory that contains expert trajectories')
        return parser
