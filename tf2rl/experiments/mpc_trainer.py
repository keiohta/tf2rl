import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from cpprb import ReplayBuffer

from tf2rl.experiments.trainer import Trainer
from tf2rl.misc.get_replay_buffer import get_space_size


class MLP(tf.keras.Model):
    def __init__(self, input_dim, output_dim, units=[32, 32],
                 name="MLP", gpu=0):
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation="relu")
        self.l2 = Dense(units[1], name="L2", activation="relu")
        self.l3 = Dense(output_dim, name="L3", activation="linear")

        with tf.device(self.device):
            self(tf.constant(np.zeros(shape=(1, input_dim), dtype=np.float32)))

    @tf.function
    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        return self.l3(features)

    def predict(self, inputs):
        assert isinstance(inputs, np.ndarray)
        if inputs.ndim == 1:
            inputs = np.expand_dims(inputs, axis=0)

        with tf.device(self.device):
            outputs = self.call(inputs)

        if inputs.shape[0] == 1:
            return outputs.numpy()[0]
        else:
            return outputs.numpy()


class RandomPolicy:
    def __init__(self, max_action, act_dim):
        self._max_action = max_action
        self._act_dim = act_dim
        self.policy_name = "RandomPolicy"

    def get_action(self):
        return np.random.uniform(
            low=-self._max_action,
            high=self._max_action,
            size=self._act_dim)

    def get_actions(self, batch_size):
        return np.random.uniform(
            low=-self._max_action,
            high=self._max_action,
            size=(batch_size, self._act_dim))


class MPCTrainer(Trainer):
    def __init__(
            self,
            policy,
            env,
            args,
            reward_fn,
            buffer_size=int(1e6),
            lr=0.001,
            **kwargs):
        super().__init__(policy, env, args, **kwargs)

        # Prepare buffer that stores transitions (s, a, s')
        rb_dict = {
            "size": buffer_size,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {
                    "shape": get_space_size(self._env.observation_space)},
                "next_obs": {
                    "shape": get_space_size(self._env.observation_space)},
                "act": {
                    "shape": get_space_size(self._env.action_space)}}}
        self.dynamics_buffer = ReplayBuffer(**rb_dict)

        # Dynamics model
        obs_dim = self._env.observation_space.high.size
        act_dim = self._env.action_space.high.size
        self._dynamics_model = MLP(
            input_dim=obs_dim + act_dim,
            output_dim=obs_dim,
            gpu=args.gpu)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Reward function
        self._reward_fn = reward_fn

    def _set_check_point(self, model_dir):
        # Save and restore model
        if isinstance(self._policy, tf.keras.Model):
            super()._set_check_point(model_dir)

    def __call__(self):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        # Gather dataset of random trajectories
        self.logger.info("Ramdomly collect {} samples...".format(self._n_random_rollout * self._episode_max_steps))
        self.collect_sample_randomly()

        for i in range(self._max_iter):
            # Train dynamics f(s, a) according to eq.(2)
            self.fit_dynamics(i)

            total_rew = 0.

            # Collect new sample
            obs = self._env.reset()
            for _ in range(self._episode_max_steps):
                total_steps += 1
                act = self._mpc(obs)
                next_obs, rew, done, _ = self._env.step(act)
                self.dynamics_buffer.add(
                    obs=obs, act=act, next_obs=next_obs)
                total_rew += rew
                if done:
                    break
                obs = next_obs
            tf.summary.experimental.set_step(total_steps)
            tf.summary.scalar("mpc/total_rew", total_rew)
            self.logger.info("iter={0: 3d} total_rew: {1:4.4f}".format(i, total_rew))

    def _mpc(self, obs):
        init_actions = self._policy.get_actions(
            batch_size=self._n_sample)
        total_rewards = np.zeros(shape=(self._n_sample,))
        obses = np.tile(obs, (self._n_sample, 1))

        for i in range(self._horizon):
            if i == 0:
                acts = init_actions
            else:
                acts = self._policy.get_actions(
                    batch_size=self._n_sample)
            assert obses.shape[0] == acts.shape[0]
            obs_diffs = self._dynamics_model.predict(
                np.concatenate([obses, acts], axis=1))
            assert obses.shape == obs_diffs.shape
            next_obses = obses + obs_diffs
            rewards = self._reward_fn(obses, acts)
            assert rewards.shape == total_rewards.shape
            total_rewards += rewards
            obses = next_obses

        idx = np.argmax(total_rewards)
        return init_actions[idx]

    def _set_from_args(self, args):
        super()._set_from_args(args)
        self._max_iter = args.max_iter
        self._horizon = args.horizon
        self._n_sample = args.n_sample
        self._n_random_rollout = args.n_random_rollout
        self._batch_size = args.batch_size

    def collect_sample_randomly(self):
        for _ in range(self._n_random_rollout):
            obs = self._env.reset()
            for _ in range(self._episode_max_steps):
                act = np.random.uniform(
                    low=self._env.action_space.low,
                    high=self._env.action_space.high,
                    size=self._env.action_space.shape[0])
                next_obs, _, done, _ = self._env.step(act)
                self.dynamics_buffer.add(
                    obs=obs, act=act, next_obs=next_obs)
                obs = next_obs
                if done:
                    break

    @tf.function
    def _fit_dynamics_body(self, inputs, labels):
        with tf.GradientTape() as tape:
            predicts = self._dynamics_model(inputs)
            loss = tf.reduce_mean(0.5 * tf.square(labels - predicts))
        grads = tape.gradient(
            loss, self._dynamics_model.trainable_variables)
        self._optimizer.apply_gradients(
            zip(grads, self._dynamics_model.trainable_variables))
        return loss

    def fit_dynamics(self, n_iter, n_epoch=1):
        samples = self.dynamics_buffer.sample(
            self.dynamics_buffer.get_stored_size())
        inputs = np.concatenate([samples["obs"], samples["act"]], axis=1)
        labels = samples["next_obs"] - samples["obs"]
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat(n_epoch)
        for batch, (x, y) in enumerate(dataset):
            loss = self._fit_dynamics_body(x, y)
            self.logger.debug("batch: {} loss: {:2.6f}".format(batch, loss))
        tf.summary.scalar("mpc/model_loss", loss)
        self.logger.info("iter={0: 3d} loss: {1:2.8f}".format(n_iter, loss))

    @staticmethod
    def get_argument(parser=None):
        parser = Trainer.get_argument(parser)
        parser.add_argument('--gpu', type=int, default=0,
                            help='GPU id')
        parser.add_argument("--max-iter", type=int, default=100)
        parser.add_argument("--horizon", type=int, default=20)
        parser.add_argument("--n-sample", type=int, default=1000)
        parser.add_argument("--n-random-rollout", type=int, default=1000)
        parser.add_argument("--batch-size", type=int, default=512)
        return parser
