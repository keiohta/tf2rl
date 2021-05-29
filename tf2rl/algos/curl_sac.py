import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import LayerNormalization

from tf2rl.algos.sac import SAC
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.tools.img_tools import random_crop, center_crop


class Encoder(tf.keras.Model):
    def __init__(self,
                 obs_shape=(84, 84, 9),
                 feature_dim=50,
                 n_conv_layers=4,
                 n_conv_filters=32,
                 name="curl_encoder"):
        super().__init__(name=name)

        assert len(obs_shape) == 3
        assert obs_shape[0] == 64 or obs_shape[0] == 84

        self.convs = []
        for layer_idx in range(n_conv_layers):
            # TODO: Check padding and activation
            stride = 2 if layer_idx == 0 else 1
            self.convs.append(
                Conv2D(n_conv_filters, kernel_size=(3, 3), strides=(stride, stride), padding='valid',
                       activation='relu'))

        self.flatten = Flatten()
        self.fc = Dense(feature_dim)
        self.layer_norm = LayerNormalization()
        self.feature_dim = feature_dim

        dummy_obs = np.zeros(shape=(1,) + obs_shape, dtype=np.int)
        with tf.device("/cpu:0"):
            self(tf.constant(dummy_obs))

    def call(self, inputs):
        features = tf.divide(tf.cast(inputs, tf.float32),
                             tf.constant(255.))

        for conv in self.convs:
            features = conv(features)
        features = self.flatten(features)
        features = self.fc(features)
        features = self.layer_norm(features)
        return features


class CURLSAC(SAC):
    def __init__(self,
                 action_dim,
                 obs_shape=(84, 84, 9),
                 n_conv_layers=4,
                 n_conv_filters=32,
                 feature_dim=50,
                 tau_encoder=0.05,
                 lr_sac=1e-3,
                 lr_curl=1e-3,
                 lr_alpha=1e-4,
                 stop_q_grad=False,
                 **kwargs):
        super().__init__(state_shape=(feature_dim,),
                         action_dim=action_dim,
                         name="CURLSAC",
                         lr=lr_sac,
                         lr_alpha=lr_alpha,
                         **kwargs)
        self._tau_encoder = tau_encoder
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

        self._curl_optimizer = tf.keras.optimizers.Adam(lr=lr_curl)
        self._curl_w = tf.Variable(initial_value=tf.random.normal(shape=(feature_dim, feature_dim)),
                                   name='curl_w', dtype=tf.float32, trainable=True)
        self._stop_q_grad = stop_q_grad
        self._input_img_size = obs_shape[0]
        self.state_ndim = 3

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        obses_anchor = random_crop(states, self._input_img_size)
        next_obses_anchor = random_crop(next_states, self._input_img_size)
        obses_negative = random_crop(states, self._input_img_size)

        td_errors, actor_loss, qf_loss, logp_min, logp_max, logp_mean, curl_loss, z_anchor, z_negatives, logits = self._train_body(
            obses_anchor, obses_negative, actions, next_obses_anchor, rewards, dones, weights)

        tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_Q_loss", data=qf_loss)
        tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
        tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
        tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
        if self.auto_alpha:
            tf.summary.scalar(name=self.policy_name + "/log_ent", data=self.log_alpha)
            tf.summary.scalar(name=self.policy_name + "/logp_mean+target", data=logp_mean + self.target_alpha)
        tf.summary.scalar(name=self.policy_name + "/ent", data=self.alpha)
        tf.summary.scalar(name=self.policy_name + "/curl_loss", data=curl_loss)
        tf.summary.scalar(name=self.policy_name + "/z_anchor", data=z_anchor)
        tf.summary.scalar(name=self.policy_name + "/z_negatives", data=z_negatives)
        tf.summary.scalar(name=self.policy_name + "/logits", data=logits)

        return td_errors

    def get_action(self, state, test=False):
        state = center_crop(state, self._input_img_size)
        return super().get_action(state, test)

    @tf.function
    def _get_action_body(self, state, test):
        encoded_state = self._encoder(state)
        actions, log_pis = self.actor(encoded_state, test)
        return actions

    @tf.function
    def _train_body(self, obses_anchor, obses_negative, actions, next_obses_anchor, rewards, dones, weights):
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2
            rewards = tf.squeeze(rewards, axis=1)
            dones = tf.squeeze(dones, axis=1)

            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                obs_features = self._encoder(obses_anchor)
                next_obs_features = self._encoder(next_obses_anchor)
                next_actions, next_logps = self.actor(next_obs_features)

                next_obs_features_target = self._encoder_target(next_obses_anchor)
                next_target_q1 = tf.stop_gradient(self.qf1_target(next_obs_features_target, next_actions))
                next_target_q2 = tf.stop_gradient(self.qf2_target(next_obs_features_target, next_actions))
                min_next_target_q = tf.minimum(next_target_q1, next_target_q2)

                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * (min_next_target_q - self.alpha * next_logps))

                # Compute loss of critic Q
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

                # Compute loss of CURL
                z_anchor = obs_features
                z_negatives = self._encoder_target(obses_negative)
                # Compute similarities with bilinear products
                logits = tf.matmul(z_anchor, tf.matmul(self._curl_w, tf.transpose(z_negatives, [1, 0])))
                logits -= tf.reduce_max(logits, axis=-1, keepdims=True)  # (batch_size, batch_size)
                curl_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(tf.range(self.batch_size), logits,
                                                                    from_logits=True))  # Eq.4

            if self._stop_q_grad:
                q1_variables = self.qf1.trainable_variables
                q2_variables = self.qf2.trainable_variables
            else:
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

            curl_grads = tape.gradient(curl_loss, [self._curl_w] + self._encoder.trainable_variables)
            self._curl_optimizer.apply_gradients(
                zip(curl_grads, [self._curl_w] + self._encoder.trainable_variables))
            update_target_variables(
                self._encoder_target.weights, self._encoder.weights, self._tau_encoder)

            del tape

        return td_loss_q1 + td_loss_q2, policy_loss, td_loss_q1, tf.reduce_min(logp), tf.reduce_max(
            logp), tf.reduce_mean(
            logp), curl_loss, tf.reduce_mean(z_anchor), tf.reduce_mean(z_negatives), tf.reduce_mean(logits)

    @staticmethod
    def get_argument(parser=None):
        parser = SAC.get_argument(parser)
        parser.add_argument('--stop-q-grad', action="store_true")
        return parser


if __name__ == "__main__":
    CURLSAC(action_dim=10)
    CURLSAC(np.zeros(shape=(32, 84, 84, 9)))
