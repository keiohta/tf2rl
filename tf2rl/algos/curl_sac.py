import numpy as np
import tensorflow as tf

from tf2rl.algos.sac_ae import SACAE
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.tools.img_tools import random_crop


class CURL(SACAE):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         skip_making_decoder=True,
                         name="CURL",
                         **kwargs)
        self._curl_w = tf.Variable(initial_value=tf.random.normal(shape=(self._feature_dim, self._feature_dim)),
                                   name='curl_w', dtype=tf.float32, trainable=True)

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        obses_anchor = random_crop(states, self._input_img_size)
        next_obses_anchor = random_crop(next_states, self._input_img_size)
        obses_negative = random_crop(states, self._input_img_size)

        # Update critic
        td_errors, qf_loss = self._update_critic(
            obses_anchor, actions, next_obses_anchor, rewards, dones, weights)
        tf.summary.scalar(name=self.policy_name + "/critic_loss", data=qf_loss)
        if self._n_update % self._update_critic_target_freq == 0:
            update_target_variables(self.qf1_target.weights, self.qf1.weights, self.tau)
            update_target_variables(self.qf2_target.weights, self.qf2.weights, self.tau)

        # Update actor
        if self._n_update % self._update_actor_freq == 0:
            obs_features = self._encoder(obses_anchor)
            actor_loss, logp_min, logp_max, logp_mean, alpha_loss = self._update_actor(obs_features)
            tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
            tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
            tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
            tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
            if self.auto_alpha:
                tf.summary.scalar(name=self.policy_name + "/log_ent", data=self.log_alpha)
                tf.summary.scalar(name=self.policy_name + "/logp_mean+target", data=logp_mean + self.target_alpha)
            tf.summary.scalar(name=self.policy_name + "/ent", data=self.alpha)
            tf.summary.scalar(name=self.policy_name + "/alpha_loss", data=alpha_loss)

        # Update encoder
        curl_loss, w, z_anchor, z_negatives, logits = self._update_encoder(obses_anchor, obses_negative)
        tf.summary.scalar(name="encoder/curl_loss", data=curl_loss)
        tf.summary.scalar(name="encoder/z_anchor", data=z_anchor)
        tf.summary.scalar(name="encoder/w", data=w)
        tf.summary.scalar(name="encoder/z_negatives", data=z_negatives)
        tf.summary.scalar(name="encoder/logits", data=logits)

        return td_errors

    @tf.function
    def _update_encoder(self, obses_anchor, obses_negative):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of CURL
                z_anchor = self._encoder(obses_anchor)
                z_negatives = self._encoder_target(obses_negative)
                # Compute similarities with bilinear products
                logits = tf.matmul(z_anchor, tf.matmul(self._curl_w, tf.transpose(z_negatives, [1, 0])))
                logits -= tf.reduce_max(logits, axis=-1, keepdims=True)  # (batch_size, batch_size)
                curl_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(tf.range(self.batch_size), logits,
                                                                    from_logits=True))  # Eq.4

            curl_grads = tape.gradient(curl_loss, [self._curl_w] + self._encoder.trainable_variables)
            self._encoder_optimizer.apply_gradients(
                zip(curl_grads, [self._curl_w] + self._encoder.trainable_variables))
            update_target_variables(
                self._encoder_target.weights, self._encoder.weights, self._tau_encoder)

        return curl_loss, tf.reduce_mean(tf.abs(self._curl_w)), tf.reduce_mean(
            tf.abs(z_anchor)), tf.reduce_mean(tf.abs(z_negatives)), tf.reduce_mean(logits)


if __name__ == "__main__":
    CURL(action_dim=10)
    CURL(np.zeros(shape=(32, 84, 84, 9)))
