import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.policies.gaussian_actor import GaussianActor
from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss


class CriticV(tf.keras.Model):
    def __init__(self, state_shape, name='vf'):
        super().__init__(name=name)

        self.l1 = Dense(256, name="L1", activation='relu')
        self.l2 = Dense(256, name="L2", activation='relu')
        self.l3 = Dense(1, name="L3", activation='linear')

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        self(dummy_state)

    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1, name="values")


class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(256, name="L1", activation='relu')
        self.l2 = Dense(256, name="L2", activation='relu')
        self.l3 = Dense(1, name="L2", activation='linear')

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        self([dummy_state, dummy_action])

    def call(self, inputs):
        [states, actions] = inputs
        features = tf.concat([states, actions], axis=1)
        features = self.l1(features)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)


class SAC(OffPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="SAC",
            max_action=1.,
            lr=3e-4,
            actor_units=[256, 256],
            tau=0.005,
            scale_reward=5.,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            **kwargs):
        super().__init__(
            name=name, memory_capacity=memory_capacity,
            n_warmup=n_warmup, **kwargs)

        self.actor = GaussianActor(
            state_shape, action_dim, max_action, squash=True,
            tanh_mean=False, tanh_std=False)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.vf = CriticV(state_shape)
        self.vf_target = CriticV(state_shape)
        update_target_variables(self.vf_target.weights,
                                self.vf.weights, tau=1.)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.qf1 = CriticQ(state_shape, action_dim, name="qf1")
        self.qf2 = CriticQ(state_shape, action_dim, name="qf2")
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Set hyper-parameters
        self.tau = tau
        self.scale_reward = scale_reward

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(tf.constant(state), test)

        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state, test):
        return self.actor(state, test)[0]

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        td_errors, actor_loss, vf_loss, qf_loss, logp_min, logp_max = \
            self._train_body(states, actions, next_states,
                             rewards, done, weights)

        tf.summary.scalar(name=self.policy_name+"/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/critic_V_loss", data=vf_loss)
        tf.summary.scalar(name=self.policy_name+"/critic_Q_loss", data=qf_loss)
        tf.summary.scalar(name=self.policy_name+"/logp_min", data=logp_min)
        tf.summary.scalar(name=self.policy_name+"/logp_max", data=logp_max)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights=None):
        with tf.device(self.device):
            rewards = tf.squeeze(rewards, axis=1)
            not_done = 1. - tf.cast(done, dtype=tf.float32)

            # Update Critic
            with tf.GradientTape(persistent=True) as tape:
                current_Q1 = self.qf1([states, actions])
                current_Q2 = self.qf2([states, actions])
                vf_next_target = self.vf_target(next_states)

                target_Q = tf.stop_gradient(
                    self.scale_reward * rewards + not_done * self.discount * vf_next_target)

                td_loss1 = tf.reduce_mean(huber_loss(
                    target_Q - current_Q1, delta=self.max_grad))
                td_loss2 = tf.reduce_mean(huber_loss(
                    target_Q - current_Q2, delta=self.max_grad))

            q1_grad = tape.gradient(td_loss1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))

            del tape

            with tf.GradientTape(persistent=True) as tape:
                current_V = self.vf(states)
                sample_actions, logp = self.actor(states)

                current_Q1 = self.qf1([states, sample_actions])
                current_Q2 = self.qf2([states, sample_actions])
                current_Q = tf.minimum(current_Q1, current_Q2)

                target_V = tf.stop_gradient(current_Q - logp)
                td_errors = target_V - current_V
                vf_loss_t = tf.reduce_mean(
                    huber_loss(td_errors, delta=self.max_grad) * weights)

                # TODO: Add reguralizer
                policy_loss = tf.reduce_mean(logp - current_Q1)

            vf_grad = tape.gradient(vf_loss_t, self.vf.trainable_variables)
            self.vf_optimizer.apply_gradients(
                zip(vf_grad, self.vf.trainable_variables))
            update_target_variables(
                self.vf_target.weights, self.vf.weights, self.tau)

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            del tape

        return td_errors, policy_loss, vf_loss_t, td_loss1, tf.reduce_min(logp), tf.reduce_max(logp)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        td_erros_Q1, td_errors_Q2, td_errors_V = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.squeeze(
            np.abs(td_erros_Q1.numpy()) +
            np.abs(td_errors_Q2.numpy()) +
            np.abs(td_errors_V.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            rewards = tf.squeeze(rewards, axis=1)
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            # Compute TD errors for Q-value func
            current_Q1 = self.qf1([states, actions])
            current_Q2 = self.qf2([states, actions])
            vf_next_target = self.vf_target(next_states)

            target_Q = tf.stop_gradient(
                self.scale_reward * rewards + not_dones * self.discount * vf_next_target)

            td_errors_Q1 = target_Q - current_Q1
            td_errors_Q2 = target_Q - current_Q2

            # Compute TD errors for V-value func
            current_V = self.vf(states)
            sample_actions, logp = self.actor(states)

            current_Q1 = self.qf1([states, sample_actions])
            current_Q2 = self.qf2([states, sample_actions])
            current_Q = tf.minimum(current_Q1, current_Q2)

            target_V = tf.stop_gradient(current_Q - logp)
            td_errors_V = target_V - current_V

        return td_errors_Q1, td_errors_Q2, td_errors_V
