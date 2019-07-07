import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.gail import GAIL
from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.networks.spectral_norm_dense import SNDense


class Discriminator(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, units=[32, 32],
                 n_latent_unit=32, enable_sn=False, name="Discriminator"):
        super().__init__(name=name)

        DenseClass = SNDense if enable_sn else Dense
        self.l1 = DenseClass(units[0], name="L1", activation="relu")
        self.l2 = DenseClass(units[1], name="L2", activation="relu")
        self.l_mean = DenseClass(n_latent_unit, name="L_mean", activation="linear")
        self.l_logstd = DenseClass(n_latent_unit, name="L_std", activation="linear")
        self.l3 = DenseClass(1, name="L3", activation="sigmoid")

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        with tf.device("/cpu:0"):
            self([dummy_state, dummy_action])

    def call(self, inputs):
        # Encoder
        features = tf.concat(inputs, axis=1)
        features = self.l1(features)
        features = self.l2(features)
        means = self.l_mean(features)
        logstds = self.l_logstd(features)
        logstds = tf.clip_by_value(
            logstds, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)
        latents = means + tf.random.normal(shape=means.shape) * tf.math.exp(logstds)
        # Binary classifier
        out = self.l3(latents)
        return out, means, logstds

    def compute_reward(self, inputs):
        features = tf.concat(inputs, axis=1)
        features = self.l1(features)
        features = self.l2(features)
        means = self.l_mean(features)
        return tf.math.log(self.l3(means) + 1e-8)


class VAIL(GAIL):
    def __init__(
            self,
            state_shape,
            action_dim,
            units=[32, 32],
            n_latent_unit=32,
            lr=5e-5,
            kl_target=0.5,
            reg_param=0.,
            enable_sn=False,
            enable_gp=False,
            name="VAIL",
            **kwargs):
        """
        :param enable_sn (bool): If true, add spectral normalization in Dense layer
        :param enable_gp (bool): If true, add gradient penalty to loss function
        """
        IRLPolicy.__init__(
            self, name=name, n_training=10, **kwargs)
        self.disc = Discriminator(
            state_shape=state_shape, action_dim=action_dim,
            units=units, n_latent_unit=n_latent_unit,
            enable_sn=enable_sn)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self._kl_target = kl_target
        self._reg_param = tf.Variable(reg_param, dtype=tf.float32)
        self._step_reg_param = tf.constant(1e-5, dtype=tf.float32)
        self._enable_gp = enable_gp

    def train(self, agent_states, agent_acts, expert_states, expert_acts):
        loss, accuracy, real_kl, fake_kl, js_divergence = self._train_body(
            agent_states, agent_acts, expert_states, expert_acts)
        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/RegParam", data=self._reg_param)
        tf.summary.scalar(name=self.policy_name+"/RealLatentKL", data=real_kl)
        tf.summary.scalar(name=self.policy_name+"/FakeLatentKL", data=fake_kl)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)

    @tf.function
    def _compute_kl_latent(self, means, log_stds):
        """
        Compute KL divergence of latent spaces over standard Normal
        distribution to compute loss in eq.5.  The formulation of
        KL divergence between two normal distributions is as follows:
            ln(\sigma_2 / \sigma_1) + {(\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2} / (2 * \sigma_2^2)
        Since the target distribution is standard Normal distributions,
        we can assume `\sigma_2 = 1` and `mean_2 = 0`.
        So, the resulting equation can be computed as:
            ln(1 / \sigma_1) + (\mu_1^2 + \sigma_1^2 - 1) / 2
        """
        return tf.reduce_sum(
            -log_stds + (tf.square(means) + tf.square(tf.exp(log_stds)) - 1.) / 2.,
                axis=-1)

    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts):
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                # Compute discriminator loss
                real_logits, real_means, real_logstds = self.disc(
                    [expert_states, expert_acts])
                fake_logits, fake_means, fake_logstds = self.disc(
                    [agent_states, agent_acts])
                disc_loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                              tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))
                # Compute KL loss
                real_kl = tf.reduce_mean(
                    self._compute_kl_latent(real_means, real_logstds))
                fake_kl = tf.reduce_mean(
                    self._compute_kl_latent(fake_means, fake_logstds))
                kl_loss = 0.5 * (real_kl - self._kl_target +
                                 fake_kl - self._kl_target)
                loss = disc_loss + self._reg_param * kl_loss
                # Gradient penalty
                if self._enable_gp:
                    raise NotImplementedError

            grads = tape.gradient(loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.disc.trainable_variables))

        # Update reguralizer parameter \beta in eq.(9)
        self._reg_param.assign(tf.maximum(
            tf.constant(0., dtype=tf.float32),
            self._reg_param + self._step_reg_param * (kl_loss - self._kl_target)))

        accuracy = \
            tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. + \
            tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.
        js_divergence = self._compute_js_divergence(
            fake_logits, real_logits)
        return loss, accuracy, real_kl, fake_kl, js_divergence
