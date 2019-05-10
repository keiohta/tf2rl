import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.envs.atari_wrapper import LazyFrames
import tf2rl.misc.target_update_ops as target_update


class QFunc(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[32, 32], name="QFunc"):
        super().__init__(name=name)

        self.l1 = tf.keras.layers.Dense(units[0], name="L1")
        self.l2 = tf.keras.layers.Dense(units[1], name="L2")
        self.l3 = tf.keras.layers.Dense(action_dim, name="L3")

        with tf.device("/cpu:0"):
            self(inputs=tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float64)))

    def call(self, inputs):
        features = tf.concat(inputs, axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        return features


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
            **kwargs):
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        q_func = q_func if q_func is not None else QFunc
        # Define and initialize Q-function network
        self.q_func = q_func(state_shape, action_dim, units)
        self.q_func_target = q_func(state_shape, action_dim, units)
        self.q_func_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        for param, target_param in zip(self.q_func.weights, self.q_func_target.weights):
            target_param.assign(param)

        self._action_dim = action_dim

        # Set hyperparameters
        self.epsilon = epsilon
        self.target_replace_interval = target_replace_interval
        self.n_update = 0

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

        return td_error

    # @tf.contrib.eager.defun
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        # TODO: Enable to run with TensorFlow graph mode (currently use np.ndarray)
        with tf.device(self.device):
            not_done = 1. - done
            actions = np.asarray(np.ravel(actions), np.int32)

            with tf.GradientTape() as tape:
                current_Q = self.q_func(states)
                target_Q = np.asarray(current_Q).copy()
                target_Q[np.arange(actions.shape[0]), actions] = \
                    np.ravel(rewards + (not_done * self.discount * tf.reduce_max(self.q_func_target(next_states), keepdims=True, axis=1)))
                target_Q = tf.stop_gradient(target_Q)
                td_error = current_Q - target_Q
                q_func_loss = tf.reduce_mean(tf.square(td_error) * weights * 0.5)

            q_func_grad = tape.gradient(q_func_loss, self.q_func.trainable_variables)
            self.q_func_optimizer.apply_gradients(zip(q_func_grad, self.q_func.trainable_variables))

            self.n_update += 1
            # Update target networks
            if self.n_update % self.target_replace_interval == 0:
                target_update.update_target_variables(self.q_func_target.weights, self.q_func.weights, tau=1.)

            return td_error, q_func_loss
