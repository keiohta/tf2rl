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
            alpha=.2,
            auto_alpha=False,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            **kwargs):
        super().__init__(
            name=name, memory_capacity=memory_capacity,
            n_warmup=n_warmup, **kwargs)

        self._set_up_actor(state_shape, action_dim, actor_units, lr, max_action)
        self._setup_critic_v(state_shape, lr)
        self._setup_critic_q(state_shape, action_dim, lr)

        # Set hyper-parameters
        self.tau = tau
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.log_alpha = tf.Variable(0., dtype=tf.float32)
            self.alpha = tf.exp(self.log_alpha)
            self.target_alpha = -action_dim
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            self.alpha = alpha

    def _set_up_actor(self, state_shape, action_dim, actor_units, lr, max_action=1.):
        self.actor = GaussianActor(
            state_shape, action_dim, max_action, squash=True,
            units=actor_units, tanh_mean=False, tanh_std=False)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_q(self, state_shape, action_dim, lr):
        self.qf1 = CriticQ(state_shape, action_dim, name="qf1")
        self.qf2 = CriticQ(state_shape, action_dim, name="qf2")
        update_target_variables(self.vf_target.weights,
                                self.vf.weights, tau=1.)
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_v(self, state_shape, lr):
        self.vf = CriticV(state_shape)
        self.vf_target = CriticV(state_shape)
        update_target_variables(self.vf_target.weights,
                                self.vf.weights, tau=1.)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

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
        # TODO: Replace `done` with `dones`
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
        if self.auto_alpha:
            self.alpha = tf.exp(self.log_alpha)
            tf.summary.scalar(name=self.policy_name + "/log_ent", data=self.log_alpha)
        tf.summary.scalar(name=self.policy_name+"/ent", data=self.alpha)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            if tf.equal(tf.rank(rewards), 2) is True:
                rewards = tf.squeeze(rewards, axis=1)
            not_done = 1. - tf.cast(done, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                current_q1 = self.qf1([states, actions])
                current_q2 = self.qf2([states, actions])
                vf_next_target = self.vf_target(next_states)

                target_q = tf.stop_gradient(
                    rewards + not_done * self.discount * vf_next_target)

                td_loss_q1 = tf.reduce_mean(huber_loss(
                    target_q - current_q1, delta=self.max_grad) * weights)
                td_loss_q2 = tf.reduce_mean(huber_loss(
                    target_q - current_q2, delta=self.max_grad) * weights)  # Eq.(7)

                # Compute loss of critic V
                current_v = self.vf(states)

                sample_actions, logp, _ = self.actor(states)  # Resample actions to update V
                current_q1 = self.qf1([states, sample_actions])
                current_q2 = self.qf2([states, sample_actions])

                target_v = tf.stop_gradient(tf.minimum(current_q1, current_q2) - self.alpha * logp)
                td_errors = target_v - current_v
                td_loss_v = tf.reduce_mean(
                    huber_loss(td_errors, delta=self.max_grad) * weights)  # Eq.(5)

                # Compute loss of policy
                policy_loss = tf.reduce_mean(
                    (self.alpha * logp - current_q1) * weights)  # Eq.(12)

                # Compute loss of temperature parameter for entropy
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean(
                        (self.log_alpha * tf.stop_gradient(logp + self.target_alpha)))

            q1_grad = tape.gradient(td_loss_q1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss_q2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))

            vf_grad = tape.gradient(td_loss_v, self.vf.trainable_variables)
            self.vf_optimizer.apply_gradients(
                zip(vf_grad, self.vf.trainable_variables))
            update_target_variables(
                self.vf_target.weights, self.vf.weights, self.tau)

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(
                    zip(alpha_grad, [self.log_alpha]))

            del tape

        return td_errors, policy_loss, td_loss_v, td_loss_q1, tf.reduce_min(logp), tf.reduce_max(logp)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        td_errors = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return td_errors.numpy()

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            # Compute TD errors for Q-value func
            current_q1 = self.qf1([states, actions])
            vf_next_target = self.vf_target(next_states)

            target_q = tf.stop_gradient(
                rewards + not_dones * self.discount * vf_next_target)

            td_errors_q1 = target_q - current_q1

        return td_errors_q1

    @staticmethod
    def get_argument(parser=None):
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--auto-alpha', action="store_true")
        return parser
