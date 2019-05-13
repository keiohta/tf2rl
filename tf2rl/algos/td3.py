import numpy as np
import tensorflow as tf

from tf2rl.algos.ddpg import DDPG, Actor
from tf2rl.misc.target_update_ops import update_target_variables



class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[400, 300], name="Critic"):
        super().__init__(name=name)

        self.l1 = tf.keras.layers.Dense(units[0], name="L1")
        self.l2 = tf.keras.layers.Dense(units[1], name="L2")
        self.l3 = tf.keras.layers.Dense(1, name="L3")

        self.l4 = tf.keras.layers.Dense(units[0], name="L4")
        self.l5 = tf.keras.layers.Dense(units[1], name="L5")
        self.l6 = tf.keras.layers.Dense(1, name="L6")

        dummy_state = tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float64))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float64))
        with tf.device("/cpu:0"):
            self([dummy_state, dummy_action])

    def call(self, inputs):
        states, actions = inputs
        xu = tf.concat([states, actions], axis=1)

        x1 = tf.nn.relu(self.l1(xu))
        x1 = tf.nn.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = tf.nn.relu(self.l4(xu))
        x2 = tf.nn.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2


class TD3(DDPG):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="TD3",
            actor_update_freq=2,
            policy_noise=0.2,
            noise_clip=0.5,
            actor_units=[400, 300],
            critic_units=[400, 300],
            lr_critic=0.001,
            **kwargs):
        super().__init__(name=name, state_shape=state_shape, action_dim=action_dim,
                         actor_units=actor_units, critic_units=critic_units,
                         lr_critic=lr_critic, **kwargs)

        self.critic = Critic(state_shape, action_dim, critic_units)
        self.critic_target = Critic(state_shape, action_dim, critic_units)
        update_target_variables(self.critic_target.weights, self.critic.weights, tau=1.)
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=lr_critic)

        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

        self._actor_update_freq = tf.constant(actor_update_freq)
        self._it = tf.Variable(0)

    @tf.contrib.eager.defun
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_error1, td_error2 = self._compute_td_error_body(
                    states, actions, next_states, rewards, done)
                critic_loss = tf.reduce_mean(
                    tf.square(td_error1) * weights * 0.5 + \
                    tf.square(td_error2) * weights * 0.5)

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            actor_loss = None
            # TODO: Update actor and target networks at specified frequency
            # tf.assign(self._it, self._it+1)
            # if tf.mod(self._it, self._actor_update_freq) == 0:
            with tf.GradientTape() as tape:
                next_actions = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic([states, next_actions]))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            update_target_variables(self.critic_target.weights, self.critic.weights, self.tau)
            update_target_variables(self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, np.abs(td_error1) + np.abs(td_error2)

    def compute_td_error(self, states, actions, next_states, rewards, done):
        td_error1, td_error2 = self._compute_td_error_body(states, actions, next_states, rewards, done)
        return np.ravel(np.abs(td_error1.numpy()) + np.abs(td_error2.numpy()))

    @tf.contrib.eager.defun
    def _compute_td_error_body(self, states, actions, next_states, rewards, done):
        with tf.device(self.device):
            not_done = 1. - done

            # Get noisy action
            next_action = self.actor_target(next_states)
            noise = tf.cast(tf.clip_by_value(
                tf.random.normal(shape=tf.shape(next_action), stddev=self._policy_noise),
                -self._noise_clip, self._noise_clip), tf.float64)
            next_action = tf.clip_by_value(
                next_action + noise, -self.actor_target.max_action, self.actor_target.max_action)

            target_Q1, target_Q2 = self.critic_target([next_states, next_action])
            target_Q = tf.minimum(target_Q1, target_Q2)
            target_Q = rewards + (not_done * self.discount * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            current_Q1, current_Q2 = self.critic([states, actions])

        return current_Q1 - target_Q, current_Q2 - target_Q
