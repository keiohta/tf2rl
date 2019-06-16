import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.policies.gaussian_actor import GaussianActor
from tf2rl.policies.categorical_actor import CategoricalActor
from tf2rl.misc.huber_loss import huber_loss


class CriticV(tf.keras.Model):
    def __init__(self, state_shape, units, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation='relu')
        self.l2 = Dense(units[1], name="L2", activation='relu')
        self.l3 = Dense(1, name="L2", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32))
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
            lr_actor=3e-4,
            lr_critic=3e-4,
            name="VPG",
            **kwargs):
        super().__init__(name=name, **kwargs)
        self._is_discrete = is_discrete
        if is_discrete:
            self.actor = CategoricalActor(
                state_shape, action_dim, actor_units)
        else:
            self.actor = GaussianActor(
                state_shape, action_dim, max_action, actor_units)
        self.critic = CriticV(state_shape, critic_units)
        self._action_dim = action_dim
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)

        single_input = state.ndim == 1
        if single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
        action, log_pi = self._get_action_body(state, test)

        if single_input:
            return action[0], log_pi
        else:
            return action, log_pi

    def get_action_and_val(self, state, test=False):
        action, log_pi = self.get_action(state, test)
        # TODO: clean code
        single_input = state.ndim == 1
        if single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
        val = self.critic(state)
        if single_input:
            val = val[0]
        return action.numpy(), log_pi.numpy(), val.numpy()

    @tf.function
    def _get_action_body(self, state, test):
        return self.actor(state, test)

    def train_actor(self, states, actions, advantages, log_pis):
        actor_loss, log_probs = self._train_actor_body(states, actions, advantages, log_pis)
        tf.summary.scalar(name=self.policy_name+"/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/logp_max", data=np.max(log_probs))
        tf.summary.scalar(name=self.policy_name+"/logp_min", data=np.min(log_probs))
        tf.summary.scalar(name=self.policy_name+"/logp_mean", data=np.mean(log_probs))
        tf.summary.scalar(name=self.policy_name+"/adv_max", data=np.max(advantages))
        tf.summary.scalar(name=self.policy_name+"/adv_min", data=np.min(advantages))
        return actor_loss

    def train_critic(self, states, returns):
        critic_loss = self._train_critic_body(states, returns)
        tf.summary.scalar(name=self.policy_name+"/critic_loss", data=critic_loss)
        return critic_loss

    @tf.function
    def _train_actor_body(self, states, actions, advantages, log_pis):
        with tf.device(self.device):
            # Train policy
            with tf.GradientTape() as tape:
                log_probs = self.actor.compute_log_probs(states, actions)
                actor_loss = tf.reduce_mean(
                    -log_probs * tf.squeeze(advantages))  # + lambda * entropy
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
                td_errors = returns - current_V
                critic_loss = tf.reduce_mean(
                    huber_loss(diff=td_errors, max_grad=self.max_grad))
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

        return critic_loss
