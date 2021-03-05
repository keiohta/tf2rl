import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense


class CategoricalActor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(256, 256),
                 hidden_activation="relu", name="CategoricalActor"):
        super().__init__(name=name)
        self.action_dim = action_dim

        base_layers = []
        for cur_layer_size in units:
            cur_layer = tf.keras.layers.Dense(cur_layer_size, activation=hidden_activation)
            base_layers.append(cur_layer)

        self.base_layers = base_layers
        self.out_prob = Dense(action_dim, activation='softmax')

        self(tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def _compute_features(self, states):
        features = states

        for cur_layer in self.base_layers:
            features = cur_layer(features)

        return features

    def _compute_dist(self, states):
        """

        Args:
            states: np.ndarray or tf.Tensor
                Inputs to neural network.

        Returns:
            tfp.distributions.Categorical
                Categorical distribution whose probabilities are
                computed using softmax activation of a neural network
        """
        features = self._compute_features(states)

        probs = self.out_prob(features)
        dist = tfp.distributions.Categorical(probs)

        return dist

    def compute_prob(self, states):
        dist = self._compute_dist(states)
        return dist.logits

    def call(self, states, test=False):
        dist = self._compute_dist(states)

        if test:
            action = tf.argmax(dist.logits, axis=1)  # (size,)
        else:
            action = dist.sample()  # (size,)
        log_prob = dist.prob(action)

        return action, log_prob

    def compute_entropy(self, states):
        dist = self._compute_dist(states)
        return dist.entropy()

    def compute_log_probs(self, states, actions):
        """Compute log probabilities of state-action pairs

        Args:
            states: tf.Tensor
                Tensors of inputs to NN
            actions: tf.Tensor
                Tensors of NOT one-hot vector.
                They will be converted to one-hot vector inside this function.

        Returns:
            Log probabilities
        """
        dist = self._compute_dist(states)
        return dist.log_prob(actions)


class CategoricalActorCritic(CategoricalActor):
    def __init__(self, *args, **kwargs):
        tf.keras.Model.__init__(self)
        self.v = Dense(1, activation="linear")
        super().__init__(*args, **kwargs)

    def call(self, states, test=False):
        features = self._compute_feature(states)
        probs = self.out_prob(features)
        dist = tfp.distributions.Categorical(probs)

        if test:
            action = tf.argmax(dist.logits, axis=1)  # (size,)
        else:
            action = dist.sample()  # (size,)
        log_prob = dist.prob(action)

        v = self.v(features)

        return action, log_prob, v
