import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.envs.atari_wrapper import LazyFrames
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss


class QFunc(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[32, 32], name="QFunc", enable_dueling_dqn=False):
        super().__init__(name=name)
        self._enable_dueling_dqn = enable_dueling_dqn

        self.l1 = tf.keras.layers.Dense(units[0], name="L1")
        self.l2 = tf.keras.layers.Dense(units[1], name="L2")
        self.l3 = tf.keras.layers.Dense(action_dim, name="L3")

        if enable_dueling_dqn:
            self.l4 = tf.keras.layers.Dense(1, name="L3")

        with tf.device("/cpu:0"):
            self(inputs=tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float64)))

    def call(self, inputs):
        features = tf.concat(inputs, axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        if self._enable_dueling_dqn:
            advantages = self.l3(features)
            v_values = self.l4(features)
            q_values = v_values + (advantages - tf.reduce_mean(advantages, axis=1, keep_dims=True))
        else:
            q_values = self.l3(features)
        return q_values


class DQN(OffPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            q_func=None,
            name="DQN",
            lr=0.001,
            units=[32, 32],
            epsilon=0.1,
            n_warmup=int(1e4),
            target_replace_interval=int(5e3),
            memory_capacity=int(1e6),
            enable_double_dqn=False,
            enable_dueling_dqn=False,
            **kwargs):
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        q_func = q_func if q_func is not None else QFunc
        # Define and initialize Q-function network
        self.q_func = q_func(state_shape, action_dim, units=units, enable_dueling_dqn=enable_dueling_dqn)
        self.q_func_target = q_func(state_shape, action_dim, units=units, enable_dueling_dqn=enable_dueling_dqn)
        self.q_func_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        update_target_variables(self.q_func_target.weights, self.q_func.weights, tau=1.)

        self._action_dim = action_dim

        # Set hyperparameters
        self.epsilon = epsilon
        self.target_replace_interval = target_replace_interval
        self.n_update = 0

        # DQN variants
        self._enable_double_dqn = enable_double_dqn

    def get_action(self, state, test=False):
        if isinstance(state, LazyFrames):
            state = np.array(state)
        assert isinstance(state, np.ndarray)

        if not test and np.random.rand() < self.epsilon:
            action = np.random.randint(self._action_dim)
        else:
            state = np.expand_dims(state, axis=0).astype(np.float64)
            action = self._get_action_body(tf.constant(state))
            action = np.argmax(action)

        return action

    @tf.contrib.eager.defun
    def _get_action_body(self, state):
        return self.q_func(state)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        td_error, q_func_loss = self._train_body(
            states, actions, next_states, rewards, done, weights)

        tf.contrib.summary.scalar(name="QFuncLoss", tensor=q_func_loss, family="loss")

        # Remove following by using tf.global_step
        self.n_update += 1
        # Update target networks
        if self.n_update % self.target_replace_interval == 0:
            update_target_variables(self.q_func_target.weights, self.q_func.weights, tau=1.)

        return td_error

    @tf.contrib.eager.defun
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(states, actions, next_states, rewards, done)
                q_func_loss = tf.reduce_mean(tf.square(td_errors) * weights * 0.5)
                # q_func_loss = tf.reduce_mean(huber_loss(diff=td_errors) * weights)

            q_func_grad = tape.gradient(q_func_loss, self.q_func.trainable_variables)
            self.q_func_optimizer.apply_gradients(zip(q_func_grad, self.q_func.trainable_variables))

            return td_errors, q_func_loss

    @tf.contrib.eager.defun
    def _compute_td_error_body(self, states, actions, next_states, rewards, done):
        # TODO: Clean code
        not_done = 1. - tf.cast(done, dtype=tf.float64)
        actions = tf.cast(actions, dtype=tf.int32)
        with tf.device(self.device):
            indices = tf.concat(
                values=[tf.expand_dims(tf.range(self.batch_size), axis=1),
                        actions], axis=1)
            current_Q = tf.expand_dims(
                tf.gather_nd(self.q_func(states), indices), axis=1)

            if self._enable_double_dqn:
                max_q_indexes = tf.argmax(self.q_func(next_states),
                                          axis=1, output_type=tf.int32)
                # TODO: Reuse predefined `indices`
                indices = tf.concat(
                    values=[tf.expand_dims(tf.range(self.batch_size), axis=1),
                            tf.expand_dims(max_q_indexes, axis=1)], axis=1)
                target_Q = tf.expand_dims(
                    tf.gather_nd(self.q_func_target(next_states), indices), axis=1)
                target_Q = rewards + not_done * self.discount * target_Q
            else:
                target_Q = rewards + not_done * self.discount * tf.reduce_max(
                    self.q_func_target(next_states), keepdims=True, axis=1)
            target_Q = tf.stop_gradient(target_Q)
            td_errors = current_Q - target_Q
        return td_errors

    @staticmethod
    def get_argument(parser=None):
        import argparse
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--enable-double-dqn', action='store_true')
        parser.add_argument('--enable-dueling-dqn', action='store_true')
        return parser
