import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.algos.models import CategoricalActor, GaussianActor
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
        action, log_pi = self._get_action_body(tf.constant(state), test)

        action = action.numpy()
        log_pi = log_pi.numpy() if not test else None
        if single_input:
            action = action[0]
            log_pi = log_pi[0] if not test else None
        return action, log_pi if not test else log_pi

    @tf.function
    def _get_action_body(self, state, test):
        if not test:
            if self._is_discrete:
                probs = self.actor(state)
                elems = tf.range(self._action_dim)
                samples = tf.multinomial(tf.log(probs), 1)
                return elems[tf.cast(samples[0][0], tf.int32)]
            else:
                return self.actor(state)
        else:
            if self._is_discrete:
                return tf.argmax(self.actor(state)) 
            else:
                return self.actor.mean_action(state), None

    def train_actor(self, states, actions, next_states, rewards, dones, log_pis):
        loss = self._train_actor_body(states, actions, next_states, rewards, dones, log_pis)
        tf.summary.scalar(name=self.policy_name+"/actor_loss", data=loss)
        return loss

    def train_critic(self, states, actions, next_states, rewards, dones):
        loss = self._train_critic_body(states, actions, next_states, rewards, dones)
        tf.summary.scalar(name=self.policy_name+"/critic_loss", data=loss)
        return loss

    @tf.function
    def _train_actor_body(self, states, actions, next_states, rewards, dones, log_pis):
        with tf.device(self.device):
            # Train policy
            with tf.GradientTape() as tape:
                not_dones = 1. - dones
                log_probs = self.actor.compute_log_probs(states, actions)
                log_likelihood_ratio = log_probs - log_pis  # old log_probs
                advantages = rewards + not_dones * self.discount * self.critic(next_states) - self.critic(states)
                actor_loss = tf.reduce_mean(-log_likelihood_ratio * advantages)  # + lambda * entropy

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            if self._is_discrete:
                raise NotImplementedError
            else:
                return actor_loss

    @tf.function
    def _train_critic_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            # Train baseline
            with tf.GradientTape() as tape:
                not_dones = 1. - dones
                target_V = self.critic(next_states)
                target_V = rewards + not_dones * self.discount * target_V
                # target_V = tf.stop_gradient(target_V)
                current_V = self.critic(states)
                td_errors = target_V - current_V
                critic_loss = tf.reduce_mean(huber_loss(diff=td_errors, max_grad=self.max_grad))
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        return critic_loss
