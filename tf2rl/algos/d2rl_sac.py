import tensorflow as tf
import tensorflow_probability as tfp

from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.policies.tfp_gaussian_actor import GaussianActor
from tf2rl.algos.sac import SAC, CriticQ, CriticV


class DenseCriticV(CriticV):
    def call(self, states):
        features = states
        for layer_idx, cur_layer in enumerate(self.base_layers):
            features = cur_layer(features)
            # Do not concatenate for the last layer
            if not layer_idx + 1 == len(self.base_layers):
                features = tf.concat((features, states), axis=1)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)


class DenseCriticQ(CriticQ):
    def call(self, states, actions):
        state_action_pairs = tf.concat((states, actions), axis=1)
        features = state_action_pairs
        for layer_idx, cur_layer in enumerate(self.base_layers):
            features = cur_layer(features)
            # Do not concatenate for the last layer
            if not layer_idx + 1 == len(self.base_layers):
                features = tf.concat((features, state_action_pairs), axis=1)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)


class DenseGaussianActor(GaussianActor):
    def _compute_dist(self, states):
        features = states
        for layer_idx, cur_layer in enumerate(self.base_layers):
            features = cur_layer(features)
            # Do not concatenate for the last layer
            if not layer_idx + 1 == len(self.base_layers):
                features = tf.concat((features, states), axis=1)
        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[mean.shape[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))


class D2RLSAC(SAC):
    def __init__(self, *args, **kwargs):
        kwargs["name"] = "D2RL_SAC"
        super().__init__(*args, **kwargs)

    def _setup_critic_q(self, state_shape, action_dim, critic_units, lr):
        self.qf1 = DenseCriticQ(state_shape, action_dim, critic_units, name="qf1")
        self.qf2 = DenseCriticQ(state_shape, action_dim, critic_units, name="qf2")
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_v(self, state_shape, critic_units, lr):
        self.vf = DenseCriticV(state_shape, critic_units)
        self.vf_target = DenseCriticV(state_shape, critic_units)
        update_target_variables(self.vf_target.weights, self.vf.weights, tau=1.)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_actor(self, state_shape, action_dim, actor_units, lr, max_action=1.):
        self.actor = DenseGaussianActor(
            state_shape, action_dim, max_action, squash=True, units=actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
