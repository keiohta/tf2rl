import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.distributions.diagonal_gaussian import DiagonalGaussian


class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action,
                 units=[256, 256], hidden_activation="relu",
                 tanh_mean=False, tanh_std=False,
                 fix_std=False, const_std=0.1,
                 state_independent_std=False,
                 squash=False, name='GaussianPolicy'):
        super().__init__(name=name)
        self.dist = DiagonalGaussian(dim=action_dim)
        self._fix_std = fix_std
        self._tanh_std = tanh_std
        self._const_std = const_std
        self._max_action = max_action
        self._squash = squash
        self._state_independent_std = state_independent_std

        self.l1 = Dense(units[0], name="L1", activation=hidden_activation)
        self.l2 = Dense(units[1], name="L2", activation=hidden_activation)
        self.out_mean = Dense(action_dim, name="L_mean",
                              activation='tanh' if tanh_mean else None)
        if not self._fix_std:
            if self._state_independent_std:
                self.out_log_std = tf.Variable(
                    initial_value=-0.5*np.ones(action_dim, dtype=np.float32),
                    dtype=tf.float32, name="logstd")
            else:
                activation = 'tanh' if tanh_std else None
                self.out_log_std = Dense(
                    action_dim, name="L_sigma", activation=activation)

        self(tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def _compute_dist(self, states):
        """
        Compute multivariate normal distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs mean and standard deviation to compute the distribution
        :return (Dict): Multivariate normal distribution
        """
        features = self.l1(states)
        features = self.l2(features)
        mean = self.out_mean(features)
        if self._fix_std:
            log_std = tf.ones_like(mean) * tf.math.log(self._const_std)
        else:
            if self._state_independent_std:
                log_std = tf.tile(
                    input=tf.expand_dims(self.out_log_std, axis=0),
                    multiples=[mean.shape[0], 1])
            else:
                log_std = self.out_log_std(features)
                if self._tanh_std:
                    log_std = self.LOG_SIG_CAP_MIN + 0.5 * \
                        (self.LOG_SIG_CAP_MAX - self.LOG_SIG_CAP_MIN) * (log_std + 1)
                else:
                    log_std = tf.clip_by_value(
                        log_std, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        return {"mean": mean, "log_std": log_std}

    def call(self, states, test=False):
        """
        Compute actions and log probabilities of the selected action
        """
        param = self._compute_dist(states)
        if test:
            raw_actions = param["mean"]
        else:
            raw_actions = self.dist.sample(param)
        logp_pis = self.dist.log_likelihood(raw_actions, param)

        actions = raw_actions

        if self._squash:
            actions = tf.tanh(raw_actions)
            logp_pis = self._squash_correction(logp_pis, actions)

        return actions * self._max_action, logp_pis

    def compute_log_probs(self, states, actions):
        actions /= self._max_action
        param = self._compute_dist(states)
        logp_pis = self.dist.log_likelihood(actions, param)
        if self._squash:
            logp_pis = self._squash_correction(logp_pis, actions)
        return logp_pis

    def compute_entropy(self, states):
        param = self._compute_dist(states)
        return self.dist.entropy(param)

    def _squash_correction(self, logp_pis, actions):
        # assert_op = tf.Assert(tf.less_equal(tf.reduce_max(actions), 1.), [actions])
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        # with tf.control_dependencies([assert_op]):
        diff = tf.reduce_sum(
            tf.math.log(1. - actions ** 2 + self.EPS), axis=1)
        return logp_pis - diff
