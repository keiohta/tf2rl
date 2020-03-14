import numpy as np
import tensorflow as tf

from tf2rl.distributions.base import Distribution


class DiagonalGaussian(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl(self, old_param, new_param):
        """
        Compute KL divergence of two distributions as:
            {(\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2} / (2 * \sigma_2^2) + ln(\sigma_2 / \sigma_1)

        :param old_param (Dict):
            Gaussian distribution to compare with that contains
            means: (batch_size * output_dim)
            std: (batch_size * output_dim)
        :param new_param (Dict): Same contents with old_param
        """
        old_means, old_log_stds = old_param["mean"], old_param["log_std"]
        new_means, new_log_stds = new_param["mean"], new_param["log_std"]
        old_std = tf.math.exp(old_log_stds)
        new_std = tf.math.exp(new_log_stds)

        numerator = tf.math.square(old_means - new_means) \
            + tf.math.square(old_std) - tf.math.square(new_std)
        denominator = 2 * tf.math.square(new_std) + 1e-8
        return tf.math.reduce_sum(numerator / denominator + new_log_stds - old_log_stds)

    def likelihood_ratio(self, x, old_param, new_param):
        llh_new = self.log_likelihood(x, new_param)
        llh_old = self.log_likelihood(x, old_param)
        return tf.math.exp(llh_new - llh_old)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
            TODO: write equation
        """
        means = param["mean"]
        log_stds = param["log_std"]
        assert means.shape == log_stds.shape
        zs = (x - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, axis=-1) \
               - 0.5 * tf.reduce_sum(tf.square(zs), axis=-1) \
               - 0.5 * self.dim * tf.math.log(2 * np.pi)

    def sample(self, param):
        means = param["mean"]
        log_stds = param["log_std"]
        # reparameterization
        return means + tf.random.normal(shape=means.shape) * tf.math.exp(log_stds)

    def entropy(self, param):
        log_stds = param["log_std"]
        return tf.reduce_sum(log_stds + tf.math.log(tf.math.sqrt(2 * np.pi * np.e)), axis=-1)
