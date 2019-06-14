import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.distributions.diagonal_gaussian import DiagonalGaussian


class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2
    LOG_SIG_CAP_MIN = -20
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action,
                 units=[256, 256], name='GaussianPolicy'):
        super().__init__(name=name)
        self.dist = DiagonalGaussian(dim=action_dim)

        self.l1 = Dense(units[0], name="L1", activation='relu')
        self.l2 = Dense(units[1], name="L2", activation='relu')
        self.out_mean = Dense(action_dim, name="L_mean")
        self.out_log_sigma = Dense(action_dim, name="L_sigma")

        self._max_action = max_action

        self(tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def _compute_dist(self, states):
        """Compute multivariate normal distribution

        Args:
            states: Inputs to neural network. NN outputs mean and
                    standard deviation to compute the distribution
        Return:
            Multivariate normal distribution
        """
        features = self.l1(states)
        features = self.l2(features)

        mean = self.out_mean(features)
        log_sigma = self.out_log_sigma(features)
        log_sigma = tf.clip_by_value(
            log_sigma, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        return {"mean":mean, "log_std":log_sigma}

    def call(self, states, test=False):
        """Compute actions and log probabilities of the selected action
        """
        param = self._compute_dist(states)
        if test:
            raw_actions = param["mean"]
        else:
            raw_actions = self.dist.sample(param)
        log_pis = self.dist.log_likelihood(raw_actions, param)

        actions = tf.tanh(raw_actions) * self._max_action

        # for variable replacement
        diff = tf.reduce_sum(
            tf.math.log(1. - actions ** 2 + self.EPS), axis=1)
        log_pis -= diff

        return actions, log_pis

    def compute_log_probs(self, states, actions):
        param = self._compute_dist(states)
        log_pis = self.dist.log_likelihood(actions, param)
        # TODO: This is correct?
        diff = tf.reduce_sum(
            tf.math.log(1. - actions ** 2 + self.EPS), axis=1)
        log_pis -= diff
        return log_pis
