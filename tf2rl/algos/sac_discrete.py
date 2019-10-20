import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.sac import CriticV, SAC
from tf2rl.policies.categorical_actor import CategoricalActor
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss


class CriticQ(tf.keras.Model):
    """
    The output of Q-function moves
        from Q: S x A -> R
        to   Q: S -> R^|A|
    compared with continuous version of SAC
    """
    def __init__(self, state_shape, action_dim, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(256, name="L1", activation='relu')
        self.l2 = Dense(256, name="L2", activation='relu')
        self.l3 = Dense(action_dim, name="L2", activation='linear')

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        self(dummy_state)

    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        values = self.l3(features)

        return values


class SACDiscrete(SAC):
    def __init__(
            self,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

    def _set_up_actor(self, state_shape, action_dim, actor_units, lr, max_action=1.):
        # The output of actor is categorical distribution
        self.actor = CategoricalActor(
            state_shape, action_dim)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_q(self, state_shape, action_dim, lr):
        self.qf1 = CriticQ(state_shape, action_dim, name="qf1")
        self.qf2 = CriticQ(state_shape, action_dim, name="qf2")
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights=None):
        with tf.device(self.device):
            rewards = tf.squeeze(rewards, axis=1)
            not_done = 1. - tf.cast(done, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                current_Q1 = self.qf1([states, actions])
                current_Q2 = self.qf2([states, actions])
                vf_next_target = self.vf_target(next_states)

                target_Q = tf.stop_gradient(
                    self.scale_reward * rewards + not_done * self.discount * vf_next_target)

                td_loss1 = tf.reduce_mean(huber_loss(
                    target_Q - current_Q1, delta=self.max_grad))
                td_loss2 = tf.reduce_mean(huber_loss(
                    target_Q - current_Q2, delta=self.max_grad))  # Eq.(7)

            q1_grad = tape.gradient(td_loss1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))

            del tape

            with tf.GradientTape(persistent=True) as tape:
                current_V = self.vf(states)
                sample_actions, logp, param = self.actor(states)

                current_Q1 = self.qf1([states, sample_actions])
                current_Q2 = self.qf2([states, sample_actions])
                current_Q = tf.minimum(current_Q1, current_Q2, axis=1)

                target_V = tf.stop_gradient(
                    tf.transpose(param["probs"]) * (current_Q - logp))
                td_errors = target_V - current_V
                vf_loss_t = tf.reduce_mean(
                    huber_loss(td_errors, delta=self.max_grad) * weights)  # Eq.(10)

                # TODO: Add reguralizer
                policy_loss = tf.reduce_mean(logp - current_Q1)  # Eq.(12)

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
