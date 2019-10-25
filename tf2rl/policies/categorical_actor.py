import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.distributions.categorical import Categorical


class CategoricalActor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[256, 256],
                 name="CategoricalActor"):
        super().__init__(name=name)
        self.dist = Categorical(dim=action_dim)
        self.action_dim = action_dim

        self.l1 = Dense(units[0], activation='relu')
        self.l2 = Dense(units[1], activation='relu')
        self.prob = Dense(action_dim, activation='softmax')

        self(tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def _compute_feature(self, states):
        features = self.l1(states)
        return self.l2(features)

    def _compute_dist(self, states):
        """
        Compute categorical distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs probabilities of K classes
        :return: Categorical distribution
        """
        features = self._compute_feature(states)
        probs = self.prob(features)
        return {"prob": probs}

    def call(self, states, test=False):
        """
        Compute actions and log probability of the selected action

        :return action (tf.Tensors): Tensor of actions
        :return log_probs (tf.Tensor): Tensors of log probabilities of selected actions
        """
        param = self._compute_dist(states)
        if test:
            action = tf.math.argmax(param["prob"], axis=1)  # (size,)
        else:
            action = tf.squeeze(self.dist.sample(param), axis=1)  # (size,)
        log_prob = self.dist.log_likelihood(
            tf.one_hot(indices=action, depth=self.action_dim), param)

        return action, log_prob, param

    def compute_entropy(self, states):
        param = self._compute_dist(states)
        return self.dist.entropy(param)

    def compute_log_probs(self, states, actions):
        """Compute log probabilities of inputted actions

        :param states (tf.Tensor): Tensors of inputs to NN
        :param actions (tf.Tensor): Tensors of NOT one-hot vector.
            They will be converted to one-hot vector inside this function.
        """
        param = self._compute_dist(states)
        actions = tf.one_hot(
            indices=tf.squeeze(actions),
            depth=self.action_dim)
        param["prob"] = tf.cond(
            tf.math.greater(tf.rank(actions), tf.rank(param["prob"])),
            lambda: tf.expand_dims(param["prob"], axis=0),
            lambda: param["prob"])
        actions = tf.cond(
            tf.math.greater(tf.rank(param["prob"]), tf.rank(actions)),
            lambda: tf.expand_dims(actions, axis=0),
            lambda: actions)
        log_prob = self.dist.log_likelihood(actions, param)
        return log_prob


class CategoricalActorCritic(CategoricalActor):
    def __init__(self, *args, **kwargs):
        tf.keras.Model.__init__(self)
        self.v = Dense(1, activation="linear")
        super().__init__(*args, **kwargs)

    def call(self, states, test=False):
        features = self._compute_feature(states)
        probs = self.prob(features)
        param = {"prob": probs}
        if test:
            action = tf.math.argmax(param["prob"], axis=1)  # (size,)
        else:
            action = tf.squeeze(self.dist.sample(param), axis=1)  # (size,)

        log_prob = self.dist.log_likelihood(
            tf.one_hot(indices=action, depth=self.action_dim), param)
        v = self.v(features)

        return action, log_prob, v
