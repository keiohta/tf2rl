import tensorflow as tf

from tf2rl.algos.ddpg import DDPG, Actor, Critic
from tf2rl.misc.target_update_ops import update_target_variables


class TD3(DDPG):
    def __init__(
            self,
            state_dim,
            action_dim,
            policy_noise=0.2,
            noise_clip=0.5,
            actor_units=[400, 300],
            critic_units=[400, 300],
            **kwargs):
        super().__init__(name="TD3", state_dim=state_dim, action_dim=action_dim,
                         actor_units=actor_units, critic_units=critic_units, **kwargs)
        self.critic_1 = self.critic
        self.critic_target_1 = self.critic_target

        self.critic_2 = Critic(state_dim, action_dim, critic_units)        
        self.critic_target_2 = Critic(state_dim, action_dim, critic_units)
        update_target_variables(self.critic_2.weights, self.critic_target_2.weights, tau=1.)

        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

    @tf.contrib.eager.defun
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            not_done = 1. - done

            with tf.GradientTape() as tape:
                next_action = self.actor_target(next_states)
                noise = tf.cast(tf.clip_by_value(
                    tf.random.normal(shape=tf.shape(next_action), stddev=self._policy_noise),
                    -self._noise_clip, self._noise_clip), tf.float64)
                next_action = tf.clip_by_value(
                    next_action + noise, -self.actor_target.max_action, self.actor_target.max_action)
                target_Q_1 = self.critic_target_1(
                    [next_states, next_action], self.device)
                target_Q_2 = self.critic_target_2(
                    [next_states, next_action], self.device)
                target_Q = tf.minimum(target_Q_1, target_Q_2)
                target_Q = rewards + (not_done * self.discount * target_Q)
                target_Q = tf.stop_gradient(target_Q)
                current_Q_1 = self.critic_1([states, actions], device=self.device)
                current_Q_2 = self.critic_1([states, actions], device=self.device)
                td_error = (current_Q_1 - target_Q) + (current_Q_2 - target_Q)
                critic_loss = tf.reduce_mean(tf.square(td_error * weights) * 0.5)

            trainable_variables = self.critic_1.trainable_variables + self.critic_2.trainable_variables
            critic_grad = tape.gradient(critic_loss, trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, trainable_variables))

            with tf.GradientTape() as tape:
                next_actions = self.actor(states, device=self.device)
                actor_loss = -tf.reduce_mean(self.critic([states, next_actions], device=self.device))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            update_target_variables(self.critic_target_1.weights, self.critic_1.weights, self.tau)
            update_target_variables(self.critic_target_2.weights, self.critic_2.weights, self.tau)
            update_target_variables(self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, td_error
