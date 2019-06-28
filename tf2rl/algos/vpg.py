import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.policies.gaussian_actor import GaussianActor
from tf2rl.policies.categorical_actor import CategoricalActor


class CriticV(tf.keras.Model):
    def __init__(self, state_shape, units, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation='relu')
        self.l2 = Dense(units[1], name="L2", activation='relu')
        self.l3 = Dense(1, name="L2", activation='linear')

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        with tf.device('/cpu:0'):
            self(dummy_state)

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)


class VPG(OnPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            is_discrete,
            max_action=1.,
            actor_units=[256, 256],
            critic_units=[256, 256],
            lr_actor=1e-3,
            lr_critic=3e-3,
            fix_std=False,
            tanh_std=False,
            const_std=0.3,
            name="VPG",
            **kwargs):
        super().__init__(name=name, **kwargs)
        self._is_discrete = is_discrete
        if is_discrete:
            self.actor = CategoricalActor(
                state_shape, action_dim, actor_units)
        else:
            self.actor = GaussianActor(
                state_shape, action_dim, max_action, actor_units,
                fix_std=fix_std, tanh_std=tanh_std, const_std=const_std)
        self.critic = CriticV(state_shape, critic_units)
        self._action_dim = action_dim
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_critic)

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)

        single_input = state.ndim == 1
        if single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
        action, logp_pi = self._get_action_body(state, test)

        if single_input:
            return action.numpy()[0], logp_pi.numpy()
        else:
            return action.numpy(), logp_pi.numpy()

    def get_action_and_val(self, state, test=False):
        single_input = state.ndim == 1
        if single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
        action, logp_pi = self.get_action(state, test)
        val = self.critic(state)
        if single_input:
            val = val[0]
            action = action[0]
        return action, logp_pi, val.numpy()

    @tf.function
    def _get_action_body(self, state, test):
        return self.actor(state, test)

    def train_actor(self, states, actions, advantages, logp_olds):
        actor_loss, log_probs = self._train_actor_body(
            states, actions, advantages)
        tf.summary.scalar(name=self.policy_name+"/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/logp_max",
                          data=np.max(log_probs))
        tf.summary.scalar(name=self.policy_name+"/logp_min",
                          data=np.min(log_probs))
        tf.summary.scalar(name=self.policy_name+"/logp_mean",
                          data=np.mean(log_probs))
        tf.summary.scalar(name=self.policy_name+"/adv_max",
                          data=np.max(advantages))
        tf.summary.scalar(name=self.policy_name+"/adv_min",
                          data=np.min(advantages))
        # TODO: Compute KL divergence and output it
        return actor_loss

    def train_critic(self, states, returns):
        critic_loss = self._train_critic_body(states, returns)
        tf.summary.scalar(name=self.policy_name +
                          "/critic_loss", data=critic_loss)
        return critic_loss

    @tf.function
    def _train_actor_body(self, states, actions, advantages):
        with tf.device(self.device):
            # Train policy
            with tf.GradientTape() as tape:
                log_probs = self.actor.compute_log_probs(states, actions)
                weights = tf.stop_gradient(tf.squeeze(advantages))
                # + lambda * entropy
                actor_loss = tf.reduce_mean(-log_probs * weights)
            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

        return actor_loss, log_probs

    @tf.function
    def _train_critic_body(self, states, returns):
        with tf.device(self.device):
            # Train baseline
            with tf.GradientTape() as tape:
                current_V = self.critic(states)
                td_errors = tf.squeeze(returns) - current_V
                critic_loss = tf.reduce_mean(0.5 * tf.square(td_errors))
            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

        return critic_loss
