import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp


class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2
    LOG_SIG_CAP_MIN = -20
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action, units=[256, 256],
                 name='GaussianPolicy'):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation='relu')
        self.l2 = Dense(units[1], name="L2", activation='relu')
        self.out_mean = Dense(action_dim, name="L_mean")
        self.out_sigma = Dense(action_dim, name="L_sigma")

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

        mu = self.out_mean(features)
        log_sigma = self.out_sigma(features)
        log_sigma = tf.clip_by_value(log_sigma, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=tf.exp(log_sigma))

    def call(self, states, test=False):
        """Compute actions and log probabilities of the selected action
        """
        dist = self._compute_dist(states)
        if test:
            raw_actions = dist.mean()
        else:
            raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        actions = tf.tanh(raw_actions) * self._max_action

        # for variable replacement
        diff = tf.reduce_sum(tf.math.log(1. - actions ** 2 + self.EPS), axis=1)
        log_pis -= diff

        actions = actions * self._max_action
        return actions, log_pis

    def compute_log_probs(self, states, actions):
        dist = self._compute_dist(states)
        log_pis = dist.log_prob(actions)
        # TODO: This is correct?
        # diff = tf.reduce_sum(tf.math.log(1. - actions ** 2 + self.EPS), axis=1)
        # log_pis -= diff
        return log_pis


class CategoricalActor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[256, 256],
                 name="CategoricalActor"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation='relu')
        self.l2 = Dense(units[1], name="L2", activation='relu')
        self.l3 = Dense(action_dim, name="L3", activation='softmax')

        self(tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def _compute_dist(self, states):
        """Compute categorical distribution

        Arg:
            states: Inputs to neural network. NN outputs probabilities
                    of K classes
        Return:
            Categorical distribution
        """
        features = self.l1(states)
        features = self.l2(features)

        probabilities = self.l3(features)
        return tfp.distributions.Categorical(
            probs=probabilities)

    def call(self, states, test=False):
        """Compute actions and log probability of the selected action

        Return:
            action: Tensors of action
            log_probs: Tensors of log probabilities of selected actions
        """
        dist = self._compute_dist(states)
        if test:
            action = tf.math.argmax(dist.probs, axis=1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
