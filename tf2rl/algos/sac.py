import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.contrib.distributions import MultivariateNormalDiag
from tensorflow.contrib.layers import xavier_initializer

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables


class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2
    LOG_SIG_CAP_MIN = -20
    EPS = 1e-6

    def __init__(self, state_dim, action_dim, max_action, units=[256, 256], name='GaussianPolicy'):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation='relu')
        self.l2 = Dense(units[1], name="L2", activation='relu')
        self.out_mean = Dense(action_dim, name="L_mean")
        self.out_sigma = Dense(action_dim, name="L_sigma")

        self._max_action = max_action

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float64))
        self(dummy_state)

    def _compute_dist(self, states):
        features = self.l1(states)
        features = self.l2(features)

        mu = self.out_mean(features)
        log_sigma = self.out_sigma(features)
        log_sigma = tf.clip_by_value(log_sigma, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        return MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))

    def call(self, states):
        dist = self._compute_dist(states)
        raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        actions = tf.tanh(raw_actions)

        # for variable replacement
        diff = tf.reduce_sum(tf.log(1. - actions ** 2 + self.EPS), axis=1)
        log_pis -= diff

        actions = actions * self._max_action
        return actions, log_pis

    def mean_action(self, states):
        dist = self._compute_dist(states)
        raw_actions = dist.mean()
        actions = tf.tanh(raw_actions) * self._max_action

        return actions


class CriticV(tf.keras.Model):
    def __init__(self, state_dim, name='vf'):
        super().__init__(name=name)

        self.l1 = Dense(256, name="L1", activation='relu')
        self.l2 = Dense(256, name="L2", activation='relu')
        self.l3 = Dense(1, name="L3", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float64))
        self(dummy_state)

    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1, name="values")


class CriticQ(tf.keras.Model):
    def __init__(self, state_dim, action_dim, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(256, name="L1", activation='relu')
        self.l2 = Dense(256, name="L2", activation='relu')
        self.l3 = Dense(1, name="L2", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float64))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float64))
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
            state_dim,
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
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        self.actor = GaussianActor(state_dim, action_dim, max_action)
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.vf = CriticV(state_dim)
        self.vf_target = CriticV(state_dim)
        update_target_variables(self.vf_target.weights, self.vf.weights, tau=1.)
        self.vf_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.qf1 = CriticQ(state_dim, action_dim, name="qf1")
        self.qf2 = CriticQ(state_dim, action_dim, name="qf2")
        self.qf1_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.qf2_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # Set hyper-parameters
        self.tau = tau
        self.scale_reward = scale_reward

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(np.float64)
        action = self._get_action_body(tf.constant(state), test)

        return action.numpy()[0]

    @tf.contrib.eager.defun
    def _get_action_body(self, state, test):
        if not test:
            return self.actor(state)[0]
        else:
            return self.actor.mean_action(state)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        td_errors, actor_loss, vf_loss, qf_loss, log_pi_min, log_pi_max = \
            self._train_body(states, actions, next_states, rewards, done, weights)

        tf.contrib.summary.scalar(name="ActorLoss", tensor=actor_loss, family="loss")
        tf.contrib.summary.scalar(name="CriticVLoss", tensor=vf_loss, family="loss")
        tf.contrib.summary.scalar(name="CriticQLoss", tensor=qf_loss, family="loss")
        tf.contrib.summary.scalar(name="log_pi_min", tensor=log_pi_min, family="loss")
        tf.contrib.summary.scalar(name="log_pi_max", tensor=log_pi_max, family="loss")

        return td_errors

    @tf.contrib.eager.defun
    def _train_body(self, states, actions, next_states, rewards, done, weights=None):
        with tf.device(self.device):
            rewards = tf.squeeze(rewards, axis=1)
            not_done = 1. - tf.cast(done, dtype=tf.float64)

            # Update Critic
            with tf.GradientTape(persistent=True) as tape:
                current_Q1 = self.qf1([states, actions])
                current_Q2 = self.qf2([states, actions])
                vf_next_target = self.vf_target(next_states)

                target_Q = tf.stop_gradient(
                    self.scale_reward * rewards + not_done * self.discount * vf_next_target)

                td_loss1 = 0.5 * tf.keras.losses.MSE(target_Q, current_Q1)
                td_loss2 = 0.5 * tf.keras.losses.MSE(target_Q, current_Q2)

            q1_grad = tape.gradient(td_loss1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(zip(q2_grad, self.qf2.trainable_variables))

            del tape

            with tf.GradientTape(persistent=True) as tape:
                current_V = self.vf(states)
                sample_actions, log_pi = self.actor(states)

                current_Q1 = self.qf1([states, sample_actions])
                current_Q2 = self.qf2([states, sample_actions])
                current_Q = tf.minimum(current_Q1, current_Q2)

                target_V = tf.stop_gradient(current_Q - log_pi)
                td_errors = target_V - current_V
                vf_loss_t = 0.5 * tf.square(td_errors) * weights

                # TODO: Add reguralizer
                policy_loss = tf.reduce_mean(log_pi - current_Q1)

            vf_grad = tape.gradient(vf_loss_t, self.vf.trainable_variables)
            self.vf_optimizer.apply_gradients(zip(vf_grad, self.vf.trainable_variables))
            update_target_variables(self.vf_target.weights, self.vf.weights, self.tau)

            actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            del tape

        return td_errors, policy_loss, vf_loss_t, td_loss1, tf.reduce_min(log_pi), tf.reduce_max(log_pi)
