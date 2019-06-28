import numpy as np
import tensorflow as tf

from tf2rl.distributions.base import Distribution


class Categorical(Distribution):
    def kl(self, old_param, new_param):
        """
        Compute the KL divergence of two Categorical distribution as:
            p_1 * (\log p_1  - \log p_2)
        """
        old_prob, new_prob = old_param["prob"], new_param["prob"]
        return tf.reduce_sum(
            old_prob * (tf.math.log(old_prob + self._tiny) - tf.math.log(new_prob + self._tiny)))

    def likelihood_ratio(self, x, old_param, new_param):
        old_prob, new_prob = old_param["prob"], new_param["prob"]
        return (tf.reduce_sum(new_prob * x) + self._tiny) / (tf.reduce_sum(old_prob * x) + self._tiny)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
            \log \sum(p_i * x_i)

        :param x (tf.Tensor or np.ndarray): Values to compute log likelihood
        :param param (Dict): Dictionary that contains probabilities of outputs
        :return (tf.Tensor): Log probabilities
        """
        probs = param["prob"]
        assert probs.shape == x.shape, \
            "Different shape inputted. You might have forgotten to convert `x` to one-hot vector."
        return tf.math.log(tf.reduce_sum(probs * x, axis=1) + self._tiny)

    def sample(self, param):
        probs = param["prob"]
        # NOTE: input to `tf.random.categorical` is log probabilities
        # For more details, see https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/random/categorical
        # [probs.shape[0], 1]
        return tf.random.categorical(tf.math.log(probs), 1)

    def entropy(self, param):
        probs = param["prob"]
        return -tf.reduce_sum(probs * tf.math.log(probs + self._tiny), axis=1)
