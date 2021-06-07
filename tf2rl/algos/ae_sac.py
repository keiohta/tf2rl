import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tf2rl.algos.sac import SAC
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.networks.dmc_model import Encoder, Decoder
from tf2rl.tools.img_tools import center_crop, preprocess_img


class AESAC(SAC):
    def __init__(self,
                 action_dim,
                 obs_shape=(84, 84, 9),
                 n_conv_layers=4,
                 n_conv_filters=32,
                 feature_dim=50,
                 tau_encoder=0.05,
                 tau_critic=0.01,
                 lr_sac=1e-3,
                 lr_encoder=1e-3,
                 lr_decoder=1e-3,
                 lr_alpha=1e-4,
                 stop_q_grad=False,
                 lambda_latent_val=1e-06,
                 decoder_weight_lambda=1e-07,
                 **kwargs):
        super().__init__(state_shape=(feature_dim,),
                         action_dim=action_dim,
                         name="AESAC",
                         lr=lr_sac,
                         lr_alpha=lr_alpha,
                         tau=tau_critic,
                         **kwargs)
        self._encoder = Encoder(obs_shape=obs_shape,
                                feature_dim=feature_dim,
                                n_conv_layers=n_conv_layers,
                                n_conv_filters=n_conv_filters,
                                name="curl_encoder")
        self._encoder_target = Encoder(obs_shape=obs_shape,
                                       feature_dim=feature_dim,
                                       n_conv_layers=n_conv_layers,
                                       n_conv_filters=n_conv_filters,
                                       name="curl_encoder_target")
        update_target_variables(self._encoder_target.weights, self._encoder.weights, tau=1.)

        self._decoder = Decoder()
        self._lambda_latent_val = lambda_latent_val
        self._encoder_optimizer = tf.keras.optimizers.Adam(lr=lr_encoder)
        self._decoder_optimizer = tfa.optimizers.AdamW(learning_rate=lr_decoder, weight_decay=decoder_weight_lambda)

        self._stop_q_grad = stop_q_grad
        self._input_img_size = obs_shape[0]
        self._tau_encoder = tau_encoder
        self.state_ndim = 3

    def get_action(self, state, test=False):
        if state.shape[:-3] != self._input_img_size:
            state = center_crop(state, self._input_img_size)
        return super().get_action(state, test)

    @tf.function
    def _get_action_body(self, state, test):
        encoded_state = self._encoder(state)
        actions, log_pis = self.actor(encoded_state, test)
        return actions

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        qf_loss, policy_loss, td_loss_q1, logp_min, logp_max, logp_mean, ae_loss, latent_vals = self._train_body(
            states, actions, next_states, rewards, dones, weights)

        tf.summary.scalar(name=self.policy_name + "/actor_loss", data=policy_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_Q_loss", data=qf_loss)
        tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
        tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
        tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
        if self.auto_alpha:
            tf.summary.scalar(name=self.policy_name + "/log_ent", data=self.log_alpha)
            tf.summary.scalar(name=self.policy_name + "/logp_mean+target", data=logp_mean + self.target_alpha)
        tf.summary.scalar(name=self.policy_name + "/ent", data=self.alpha)
        tf.summary.scalar(name=self.policy_name + "/ae_loss", data=ae_loss)
        tf.summary.scalar(name=self.policy_name + "/latent_vals", data=latent_vals)

        return qf_loss

    @tf.function
    def _train_body(self, obses, actions, next_obses, rewards, dones, weights):
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2
            rewards = tf.squeeze(rewards, axis=1)
            dones = tf.squeeze(dones, axis=1)

            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                next_obs_features = self._encoder(next_obses)
                next_actions, next_logps = self.actor(next_obs_features)

                next_obs_features_target = self._encoder_target(next_obses)
                next_target_q1 = tf.stop_gradient(self.qf1_target(next_obs_features_target, next_actions))
                next_target_q2 = tf.stop_gradient(self.qf2_target(next_obs_features_target, next_actions))
                min_next_target_q = tf.minimum(next_target_q1, next_target_q2)

                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * (min_next_target_q - self.alpha * next_logps))

                # Compute loss of critic Q
                obs_features = self._encoder(obses, stop_q_grad=self._stop_q_grad)
                current_q1 = self.qf1(obs_features, actions)
                current_q2 = self.qf2(obs_features, actions)
                td_loss_q1 = tf.reduce_mean((target_q - current_q1) ** 2)
                td_loss_q2 = tf.reduce_mean((target_q - current_q2) ** 2)  # Eq.(7)

                # Compute loss of policy
                sample_actions, logp = self.actor(obs_features)  # Resample actions to update V
                current_q1 = self.qf1(obs_features, sample_actions)
                current_q2 = self.qf2(obs_features, sample_actions)
                current_min_q = tf.minimum(current_q1, current_q2)
                policy_loss = tf.reduce_mean(self.alpha * logp - current_min_q)  # Eq.(12)

                # Compute loss of temperature parameter for entropy
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean(
                        (self.alpha * tf.stop_gradient(logp + self.target_alpha)))

                # Compute loss of AE
                rec_obses = self._decoder(obs_features)
                true_obses = preprocess_img(obses)
                rec_loss = tf.reduce_mean(tf.keras.losses.MSE(true_obses, rec_obses))
                latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.math.pow(obs_features, 2)))
                ae_loss = rec_loss + self._lambda_latent_val * latent_loss

            q1_variables = self.qf1.trainable_variables + self._encoder.trainable_variables
            q2_variables = self.qf2.trainable_variables + self._encoder.trainable_variables
            q1_grad = tape.gradient(td_loss_q1, q1_variables)
            self.qf1_optimizer.apply_gradients(zip(q1_grad, q1_variables))
            q2_grad = tape.gradient(td_loss_q2, q2_variables)
            self.qf2_optimizer.apply_gradients(zip(q2_grad, q2_variables))
            update_target_variables(self.qf1_target.weights, self.qf1.weights, self.tau)
            update_target_variables(self.qf2_target.weights, self.qf2.weights, self.tau)

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(
                    zip(alpha_grad, [self.log_alpha]))

            encoder_grads = tape.gradient(ae_loss, self._encoder.trainable_variables)
            self._encoder_optimizer.apply_gradients(zip(encoder_grads, self._encoder.trainable_variables))
            decoder_grads = tape.gradient(ae_loss, self._decoder.trainable_variables)
            self._encoder_optimizer.apply_gradients(zip(decoder_grads, self._decoder.trainable_variables))
            update_target_variables(
                self._encoder_target.weights, self._encoder.weights, self._tau_encoder)

        logp_min, logp_max, logp_mean = tf.reduce_min(logp), tf.reduce_max(logp), tf.reduce_mean(logp)
        return td_loss_q1 + td_loss_q2, policy_loss, td_loss_q1, logp_min, logp_max, logp_mean, ae_loss, tf.reduce_mean(obs_features)
