import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.distributions.diagonal_gaussian import DiagonalGaussian


class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action,
                 units=[256, 256], fix_std=False,
                 tanh_mean=True, tanh_std=True, squash=False,
                 const_std=0.1, name='GaussianPolicy'):
        super().__init__(name=name)
        self.dist = DiagonalGaussian(dim=action_dim)
        self.fix_std = fix_std
        self._tanh_std = tanh_std
        self.const_std = const_std
        self._max_action = max_action
        self._squash = squash

        self.l1 = Dense(units[0], name="L1", activation='relu')
        self.l2 = Dense(units[1], name="L2", activation='relu')
        self.out_mean = Dense(action_dim, name="L_mean",
                              activation='tanh' if tanh_mean else None)
        if not self.fix_std:
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
        if self.fix_std:
            log_std = tf.ones_like(mean) * tf.math.log(self.const_std)
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

        actions = tf.tanh(raw_actions)

        if self._squash:
            logp_pis = self._squash_correction(logp_pis, actions)

        return actions * self._max_action, logp_pis

    def compute_log_probs(self, states, actions):
        param = self._compute_dist(states)
        logp_pis = self.dist.log_likelihood(actions, param)
        if self._squash:
            logp_pis = self._squash_correction(logp_pis, actions)
        return logp_pis

    def _squash_correction(self, logp_pis, actions):
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        diff = tf.reduce_sum(
            tf.math.log(1. - actions ** 2 + self.EPS), axis=1)
        return logp_pis - diff
