import os

import numpy as np
import tensorflow as tf
from cpprb import ReplayBuffer

from tf2rl.envs.utils import is_discrete
from tf2rl.experiments.mpc_trainer import MPCTrainer
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.misc.get_replay_buffer import get_replay_buffer


class MeTrpoTrainer(MPCTrainer):
    def __init__(self, *args, n_eval_episodes_per_model=5, **kwargs):
        kwargs["n_dynamics_model"] = 5
        super().__init__(*args, **kwargs)
        self._n_eval_episodes_per_model = n_eval_episodes_per_model

        # Replay buffer to train policy
        self.replay_buffer = get_replay_buffer(self._policy, self._env)

        # Replay buffer to compute GAE
        rb_dict = {
            "size": self._episode_max_steps,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": self._env.observation_space.shape},
                "act": {"shape": self._env.action_space.shape},
                "next_obs": {"shape": self._env.observation_space.shape},
                "rew": {},
                "done": {},
                "logp": {},
                "val": {}}}
        self.local_buffer = ReplayBuffer(**rb_dict)

    def predict_next_state(self, obses, acts, idx=None):
        is_single_input = obses.ndim == acts.ndim and acts.ndim == 1
        if is_single_input:
            obses = np.expand_dims(obses, axis=0)
            acts = np.expand_dims(acts, axis=0)

        inputs = np.concatenate([obses, acts], axis=1)
        idx = np.random.randint(self._n_dynamics_model) if idx is None else idx
        obs_diffs = self._dynamics_models[idx].predict(inputs)

        if is_single_input:
            return obses[0] + obs_diffs

        return obses + obs_diffs

    def _make_inputs_output_pairs(self, n_epoch):
        samples = self.dynamics_buffer.sample(self.dynamics_buffer.get_stored_size())
        inputs = np.concatenate([samples["obs"], samples["act"]], axis=1)
        labels = samples["next_obs"] - samples["obs"]

        return inputs, labels

    def __call__(self):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)

        while True:
            # Collect (s, a, s') pairs in a real environment
            self.collect_transitions_real_env()
            total_steps += self._n_collect_steps
            tf.summary.experimental.set_step(total_steps)

            # Train dynamics models
            self.fit_dynamics(n_epoch=1)
            if self._debug:
                ret_real_env, ret_sim_env = self._evaluate_model()
                self.logger.info("Returns (real, sim) = ({: .3f}, {: .3f})".format(ret_real_env, ret_sim_env))

            # Prepare initial states for evaluation
            init_states_for_eval = np.array([
                self._env.reset() for _ in range(self._n_dynamics_model * self._n_eval_episodes_per_model)])

            # Returns to evaluate policy improvement
            returns_before_update = self._evaluate_current_return(init_states_for_eval)

            n_updates = 0
            improve_ratios = []
            while True:
                n_updates += 1

                # Generate samples using dynamics models (simulated env)
                average_return = self.collect_transitions_sim_env()

                # Update policy
                self.update_policy()

                # Evaluate policy improvement
                returns_after_update = self._evaluate_current_return(init_states_for_eval)
                n_improved = np.sum(returns_after_update > returns_before_update)
                improved_ratio = n_improved / (self._n_dynamics_model * self._n_eval_episodes_per_model)
                improve_ratios.append(improved_ratio)
                if improved_ratio < 0.7:
                    break
                returns_before_update = returns_after_update

            self.logger.info(
                "Training total steps: {0: 7} sim return: {1: .4f} n_update: {2:}, ratios: {3:}".format(
                    total_steps, average_return, n_updates, improve_ratios))
            tf.summary.scalar(name="mpc/n_updates", data=n_updates)

            # Evaluate policy in a real environment
            if total_steps // self._n_collect_steps % 10 == 0:
                avg_test_return = self.evaluate_policy(total_steps)
                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))
                tf.summary.scalar(
                    name="Common/average_test_return", data=avg_test_return)

    def _evaluate_model(self):
        ret_real_env, ret_sim_env = 0., 0.
        n_episodes = 10
        for _ in range(n_episodes):
            real_obs = self._env.reset()
            sim_obs = real_obs.copy()
            for _ in range(self._episode_max_steps):
                act, _ = self._policy.get_action(real_obs)
                if not is_discrete(self._env.action_space):
                    env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
                else:
                    env_act = act

                next_real_obs, rew, _, _ = self._env.step(env_act)
                ret_real_env += rew
                real_obs = next_real_obs

                next_sim_obs = self.predict_next_state(sim_obs, env_act)
                ret_sim_env += self._reward_fn(real_obs, act)[0]
                sim_obs = next_sim_obs

        ret_real_env /= n_episodes
        ret_sim_env /= n_episodes
        return ret_real_env, ret_sim_env

    def update_policy(self):
        # Compute mean and std for normalizing advantage
        if self._policy.normalize_adv:
            samples = self.replay_buffer.get_all_transitions()
            mean_adv = np.mean(samples["adv"])
            std_adv = np.std(samples["adv"])

        for _ in range(self._policy.n_epoch):
            samples = self.replay_buffer._encode_sample(np.random.permutation(self._policy.horizon))
            adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8) if self._policy.normalize_adv else samples["adv"]
            for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                target = slice(idx * self._policy.batch_size,
                               (idx + 1) * self._policy.batch_size)
                self._policy.train(
                    states=samples["obs"][target],
                    actions=samples["act"][target],
                    advantages=adv[target],
                    logp_olds=samples["logp"][target],
                    returns=samples["ret"][target])

    def _evaluate_current_return(self, init_states):
        n_episodes = self._n_dynamics_model * self._n_eval_episodes_per_model
        assert init_states.shape[0] == n_episodes

        obses = init_states.copy()
        next_obses = np.zeros_like(obses)
        returns = np.zeros(shape=(n_episodes,), dtype=np.float32)

        for _ in range(self._episode_max_steps):
            acts, _ = self._policy.get_action(obses)
            for i in range(n_episodes):
                model_idx = i // self._n_eval_episodes_per_model
                if not is_discrete(self._env.action_space):
                    env_act = np.clip(acts[i], self._env.action_space.low, self._env.action_space.high)
                else:
                    env_act = acts[i]
                next_obses[i] = self.predict_next_state(obses[i], env_act, idx=model_idx)
            returns += self._reward_fn(obses, acts)
            obses = next_obses

        return returns

    def _visualize_current_performance(self):
        obs = self._env.reset()
        for _ in range(self._episode_max_steps):
            act, _ = self._policy.get_action(obs)
            if not is_discrete(self._env.action_space):
                env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
            else:
                env_act = act
            next_obs = self.predict_next_state(obs, env_act)

            self._env.state = np.array([np.arctan2(next_obs[1], next_obs[0]), next_obs[2]], dtype=np.float32)
            # print(obs, act, next_obs, self._env.state)
            self._env.render()
            obs = next_obs

    def collect_transitions_real_env(self):
        total_steps = 0
        episode_steps = 0
        obs = self._env.reset()
        while total_steps < self._n_collect_steps:
            episode_steps += 1
            total_steps += 1
            act, _ = self._policy.get_action(obs)
            if not is_discrete(self._env.action_space):
                env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
            else:
                env_act = act
            next_obs, _, done, _ = self._env.step(env_act)
            self.dynamics_buffer.add(
                obs=obs, act=env_act, next_obs=next_obs)
            obs = next_obs
            if done or episode_steps == self._episode_max_steps:
                episode_steps = 0
                obs = self._env.reset()

    def collect_transitions_sim_env(self):
        """
        Generate transitions using dynamics model
        :return:
        """
        self.replay_buffer.clear()
        n_episodes = 0
        ave_episode_return = 0
        while self.replay_buffer.get_stored_size() < self._policy.horizon:
            obs = self._env.reset()
            episode_return = 0.
            for _ in range(self._episode_max_steps):
                act, logp, val = self._policy.get_action_and_val(obs)
                if not is_discrete(self._env.action_space):
                    env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
                else:
                    env_act = act
                if self._debug:
                    next_obs, rew, _, _ = self._env.step(env_act)
                else:
                    next_obs = self.predict_next_state(obs, env_act)
                    rew = self._reward_fn(obs, act)[0]
                self.local_buffer.add(obs=obs, act=act, next_obs=next_obs, rew=rew,
                                      done=False, logp=logp, val=val)
                obs = next_obs
                episode_return += rew
            self.finish_horizon(last_val=val)
            ave_episode_return += episode_return
            n_episodes += 1
        return ave_episode_return / n_episodes

    def finish_horizon(self, last_val=0):
        """
        TODO: These codes are completly identical to the ones defined in on_policy_trainer.py. Use it.
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
            for _ in range(self._episode_max_steps):
                act, _ = self._policy.get_action(obs, test=True)
                act = (act if not hasattr(self._env.action_space, "high") else
                       np.clip(act, self._env.action_space.low, self._env.action_space.high))
                next_obs, reward, done, _ = self._test_env.step(act)
                if self._save_test_path:
                    replay_buffer.add(
                        obs=obs, act=act, next_obs=next_obs,
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
            tf.summary.image('train/input_img', images, )
        return avg_test_return / self._test_episodes

    def _set_from_args(self, args):
        super()._set_from_args(args)
        self._n_collect_steps = args.n_collect_steps
        self._debug = args.debug

    @staticmethod
    def get_argument(parser=None):
        parser = MPCTrainer.get_argument(parser)
        parser.add_argument("--n-collect-steps", type=int, default=100)
        parser.add_argument("--debug", action='store_true')
        return parser
