import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from cpprb import ReplayBuffer

from tf2rl.experiments.trainer import Trainer
from tf2rl.misc.get_replay_buffer import get_space_size


class DynamicsModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, units=[32, 32], name="DymamicsModel", gpu=0):
        """
        Initialize DynamicsModel

        Args:
            input_dim (int)
            output_dim (int)
            units (iterable of int): The default is ``[32, 32]``
            name (str): The default is ``"DynamicsModel"``
            gpu (int): The default is ``0``.
        """
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation="relu")
        self.l2 = Dense(units[1], name="L2", activation="relu")
        self.l3 = Dense(output_dim, name="L3", activation="linear")

        with tf.device(self.device):
            self(tf.constant(np.zeros(shape=(1, input_dim), dtype=np.float32)))

    @tf.function
    def call(self, inputs):
        """
        Call Dynamics Model

        Args:
            inputs (tf.Tensor)

        Returns:
            tf.Tensor
        """
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
        """
        Initialize RandomPolicy

        Args:
            max_action (float)
            act_dim (int)
        """
        self._max_action = max_action
        self._act_dim = act_dim
        self.policy_name = "RandomPolicy"

    def get_action(self, obs):
        """
        Get random action

        Args:
            obs

        Returns:
            float: action
        """
        return np.random.uniform(
            low=-self._max_action,
            high=self._max_action,
            size=self._act_dim)

    def get_actions(self, obses):
        """
        Get batch actions

        Args:
            obses

        Returns:
            np.dnarray: batch actions
        """
        batch_size = obses.shape[0]
        return np.random.uniform(
            low=-self._max_action,
            high=self._max_action,
            size=(batch_size, self._act_dim))


class MPCTrainer(Trainer):
    """
    Trainer class for Model Predictive Control (MPC): https://arxiv.org/abs/1708.02596

    Command Line Args:

        * ``--max-steps`` (int): The maximum steps for training. The default is ``int(1e6)``
        * ``--episode-max-steps`` (int): The maximum steps for an episode. The default is ``int(1e3)``
        * ``--n-experiments`` (int): Number of experiments. The default is ``1``
        * ``--show-progress``: Call ``render`` function during training
        * ``--save-model-interval`` (int): Interval to save model. The default is ``int(1e4)``
        * ``--save-summary-interval`` (int): Interval to save summary. The default is ``int(1e3)``
        * ``--model-dir`` (str): Directory to restore model.
        * ``--dir-suffix`` (str): Suffix for directory that stores results.
        * ``--normalize-obs``: Whether normalize observation
        * ``--logdir`` (str): Output directory name. The default is ``"results"``
        * ``--evaluate``: Whether evaluate trained model
        * ``--test-interval`` (int): Interval to evaluate trained model. The default is ``int(1e4)``
        * ``--show-test-progress``: Call ``render`` function during evaluation.
        * ``--test-episodes`` (int): Number of episodes at test. The default is ``5``
        * ``--save-test-path``: Save trajectories of evaluation.
        * ``--show-test-images``: Show input images to neural networks when an episode finishes
        * ``--save-test-movie``: Save rendering results.
        * ``--use-prioritized-rb``: Use prioritized experience replay
        * ``--use-nstep-rb``: Use Nstep experience replay
        * ``--n-step`` (int): Number of steps for nstep experience reward. The default is ``4``
        * ``--logging-level`` (DEBUG, INFO, WARNING): Choose logging level. The default is ``INFO``
        * ``--gpu`` (int): The default is ``0``
        * ``--max-iter`` (int): Maximum iteration. The default is ``100``
        * ``--horizon`` (int): Number of steps to online horizon
        * ``--n-sample`` (int): Number of samples. The default is ``1000``
        * ``--batch-size`` (int): Batch size. The default is ``512``.
    """
    def __init__(
            self,
            policy,
            env,
            args,
            reward_fn,
            buffer_size=int(1e6),
            n_dynamics_model=1,
            lr=0.001,
            **kwargs):
        """
        Initialize MPCTrainer class

        Args:
            policy: Policy to be trained
            env (gym.Env): Environment for train
            args (Namespace or dict): config parameters specified with command line
            test_env (gym.Env): Environment for test.
            reward_fn (callable): Reward function
            buffer_size (int): The default is ``int(1e6)``
            n_dynamics_model (int): Number of dynamics models. The default is ``1``.
            lr (float): Learning rate for dynamics model. The default is ``0.001``.
        """
        super().__init__(policy, env, args, **kwargs)

        self.dynamics_buffer = ReplayBuffer(**self._prepare_dynamics_buffer_dict(buffer_size=buffer_size))
        self._n_dynamics_model = n_dynamics_model

        # Reward function
        self._reward_fn = reward_fn
        self._prepare_dynamics_model(gpu=args.gpu, lr=lr)

    def _prepare_dynamics_buffer_dict(self, buffer_size):
        # Prepare buffer that stores transitions (s, a, s')
        rb_dict = {
            "size": buffer_size,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": get_space_size(self._env.observation_space)},
                "next_obs": {"shape": get_space_size(self._env.observation_space)},
                "act": {"shape": get_space_size(self._env.action_space)}}}
        return rb_dict

    def _prepare_dynamics_model(self, gpu=0, lr=0.001):
        # Dynamics model
        obs_dim = self._env.observation_space.high.size
        act_dim = self._env.action_space.high.size
        self._dynamics_models = [
            DynamicsModel(
                input_dim=obs_dim + act_dim,
                output_dim=obs_dim,
                gpu=gpu) for _ in range(self._n_dynamics_model)]
        self._optimizers = [
            tf.keras.optimizers.Adam(learning_rate=lr) for _ in range(self._n_dynamics_model)]

    def _set_check_point(self, model_dir):
        # Save and restore model
        if isinstance(self._policy, tf.keras.Model):
            super()._set_check_point(model_dir)

    def __call__(self):
        """
        Execute Training
        """
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        # Gather dataset of random trajectories
        self.logger.info("Ramdomly collect {} samples...".format(self._n_random_rollout * self._episode_max_steps))
        self.collect_episodes(n_rollout=self._n_random_rollout)

        for i in range(self._max_iter):
            # Train dynamics f(s, a) according to eq.(2)
            mean_loss = self.fit_dynamics(n_epoch=1)

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
            self.logger.info("iter={0: 3d} total_rew: {1:4.4f} loss: {2:2.8f}".format(i, total_rew, mean_loss))

    def predict_next_state(self, obses, acts):
        """
        Predict Next State

        Args:
            obses
            acts

        Returns:
            np.ndarray: next state
        """
        obs_diffs = np.zeros_like(obses)
        inputs = np.concatenate([obses, acts], axis=1)
        for dynamics_model in self._dynamics_models:
            obs_diffs += dynamics_model.predict(inputs)
        obs_diffs /= self._n_dynamics_model
        return obses + obs_diffs

    def _mpc(self, obs):
        obses = np.tile(obs, (self._n_sample, 1))
        init_actions = self._policy.get_actions(obses)
        total_rewards = np.zeros(shape=(self._n_sample,))

        for i in range(self._horizon):
            if i == 0:
                acts = init_actions
            else:
                acts = self._policy.get_actions(obses)
            assert obses.shape[0] == acts.shape[0]
            next_obses = self.predict_next_state(obses, acts)
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

    def collect_episodes(self, n_rollout=1):
        """
        Collect Episodes

        Args:
            n_rollout (int): Number of rollout. The default is ``1``
        """
        for _ in range(n_rollout):
            obs = self._env.reset()
            for _ in range(self._episode_max_steps):
                act = self._policy.get_action(obs)
                next_obs, _, done, _ = self._env.step(act)
                self.dynamics_buffer.add(
                    obs=obs, act=act, next_obs=next_obs)
                obs = next_obs
                if done:
                    break

    @tf.function
    def _fit_dynamics_body(self, inputs, labels):
        losses = []
        for dynamics_model, optimizer in zip(self._dynamics_models, self._optimizers):
            with tf.GradientTape() as tape:
                predicts = dynamics_model(inputs)
                loss = tf.reduce_mean(0.5 * tf.square(labels - predicts))
            grads = tape.gradient(
                loss, dynamics_model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, dynamics_model.trainable_variables))
            losses.append(loss)
        return tf.convert_to_tensor(losses)

    def _make_inputs_output_pairs(self, n_epoch):
        samples = self.dynamics_buffer.sample(self.dynamics_buffer.get_stored_size())
        inputs = np.concatenate([samples["obs"], samples["act"]], axis=1)
        labels = samples["next_obs"] - samples["obs"]

        return inputs, labels

    def fit_dynamics(self, n_epoch=1):
        """
        Fit dynamics

        Args:
            n_epocs (int): Number of epocs to fit
        """
        inputs, labels = self._make_inputs_output_pairs(n_epoch)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat(n_epoch)

        mean_losses = np.zeros(shape=(self._n_dynamics_model,), dtype=np.float32)
        for batch, (x, y) in enumerate(dataset):
            _mean_losses = self._fit_dynamics_body(x, y)
            mean_losses += _mean_losses.numpy()
        mean_losses /= (batch + 1)

        for model_idx, mean_loss in enumerate(mean_losses):
            tf.summary.scalar("mpc/model_{}_loss".format(model_idx), mean_loss)
        return np.mean(mean_losses)

    @staticmethod
    def get_argument(parser=None):
        """
        Create or update argument parser for command line program

        Args:
            parser (argparse.ArgParser, optional): argument parser

        Returns:
            argparse.ArgParser: argument parser
        """
        parser = Trainer.get_argument(parser)
        parser.add_argument('--gpu', type=int, default=0,
                            help='GPU id')
        parser.add_argument("--max-iter", type=int, default=100)
        parser.add_argument("--horizon", type=int, default=20)
        parser.add_argument("--n-sample", type=int, default=1000)
        parser.add_argument("--n-random-rollout", type=int, default=1000)
        parser.add_argument("--batch-size", type=int, default=512)
        return parser
