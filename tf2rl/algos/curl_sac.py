import numpy as np
import tensorflow as tf

from tf2rl.algos.sac_ae import SACAE
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.tools.img_tools import random_crop


class CURL(SACAE):
    """
    Contrastive Unsuper Representations for Reinforcement Learning (CURL) Agent: https://arxiv.org/abs/2004.04136

    Command Line Args:

        * ``--n-warmup`` (int): Number of warmup steps before training. The default is ``1e4``.
        * ``--batch-size`` (int): Batch size of training. The default is ``32``.
        * ``--gpu`` (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        * ``--memory-capacity`` (int): Replay Buffer size. The default is ``1e5``.
        * ``--alpha`` (float): Temperature parameter. The default is ``0.2``.
        * ``--auto-alpha``: Automatic alpha tuning.
        * ``--stop-q-grad``: Whether stop gradient after convolutional layers at Encoder
    """
    def __init__(self,
                 *args,
                 **kwargs):
        """
        Initialize CURL

        Args:
            action_dim (int):
            obs_shape: (iterable of int): The default is ``(84, 84, 9)``
            n_conv_layers (int): Number of convolutional layers at encoder. The default is ``4``
            n_conv_filters (int): Number of filters in convolutional layers. The default is ``32``
            feature_dim (int): Number of features after encoder. This features are treated as SAC input. The default is ``50``
            tau_encoder (float): Target network update rate for Encoder. The default is ``0.05``
            tau_critic (float): Target network update rate for Critic. The default is ``0.01``
            auto_alpha (bool): Automatic alpha tuning. The default is ``True``
            lr_sac (float): Learning rate for SAC. The default is ``1e-3``
            lr_encoder (float): Learning rate for Encoder. The default is ``1e-3``
            lr_decoder (float): Learning rate for Decoder. The default is ``1e-3``
            update_critic_target_freq (int): The default is ``2``
            update_actor_freq (int): The default is ``2``
            lr_alpha (alpha): Learning rate for alpha. The default is ``1e-4``.
            init_temperature (float): Initial temperature. The default is ``0.1``
            stop_q_grad (bool): Whether sotp gradient propagation after encoder convolutional network. The default is ``False``
            lambda_latent_val (float): AE loss = REC loss + ``lambda_latent_val`` * latent loss. The default is ``1e-6``
            decoder_weight_lambda (float): Weight decay of AdamW for Decoder. The default is ``1e-7``
            name (str): Name of network. The default is ``"CURL"``
            max_action (float):
            actor_units (iterable of int): Numbers of units at hidden layers of actor. The default is ``(256, 256)``.
            critic_units (iterable of int): Numbers of units at hidden layers of critic. The default is ``(256, 256)``.
            alpha (float): Temperature parameter. The default is ``0.2``.
            n_warmup (int): Number of warmup steps before training. The default is ``int(1e4)``.
            memory_capacity (int): Replay Buffer size. The default is ``int(1e6)``.
            batch_size (int): Batch size. The default is ``256``.
            discount (float): Discount factor. The default is ``0.99``.
            max_grad (float): Maximum gradient. The default is ``10``.
            gpu (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        """
        super().__init__(*args,
                         skip_making_decoder=True,
                         name="CURL",
                         **kwargs)
        self._curl_w = tf.Variable(initial_value=tf.random.normal(shape=(self._feature_dim, self._feature_dim)),
                                   name='curl_w', dtype=tf.float32, trainable=True)

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        """
        Train CURL

        Args:
            states
            actions
            next_states
            rewards
            done
            weights (optional): Weights for importance sampling
        """
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
        curl_loss, w, z_anchor, logits = self._update_encoder(obses_anchor, obses_negative)
        tf.summary.scalar(name="encoder/curl_loss", data=curl_loss)
        tf.summary.scalar(name="encoder/latent_vars", data=z_anchor)
        tf.summary.scalar(name="encoder/w", data=w)
        tf.summary.scalar(name="encoder/logits", data=logits)

        self._n_update += 1

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
            tf.abs(z_anchor)), tf.reduce_mean(logits)


if __name__ == "__main__":
    CURL(action_dim=10)
    CURL(np.zeros(shape=(32, 84, 84, 9)))
