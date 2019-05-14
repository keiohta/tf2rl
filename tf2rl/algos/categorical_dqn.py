import numpy as np
import tensorflow as tf

from tf2rl.algos.dqn import DQN, QFunc
from tf2rl.envs.atari_wrapper import LazyFrames


class CategoricalQFunc(QFunc):
    def __init__(self, state_shape, action_dim, n_atoms=51, **kwargs):
        self._n_atoms = n_atoms
        self._action_dim = action_dim
        super().__init__(
            state_shape=state_shape,
            action_dim=action_dim*n_atoms,
            **kwargs)

    def call(self, inputs):
        features = tf.concat(inputs, axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        if self._enable_dueling_dqn:
            raise NotImplementedError
        else:
            features = tf.nn.relu(self.l3(features))
            features = tf.reshape(features, (-1, self._action_dim, self._n_atoms))
            return tf.clip_by_value(
                tf.keras.activations.softmax(features, axis=2),
                1e-8, 1.0-1e-8)


class CategoricalDQN(DQN):
    def __init__(self, *args, **kwargs):
        kwargs["q_func"] = CategoricalQFunc
        super().__init__(*args, **kwargs)
        self._v_max, self._v_min = 20., 0.
        self._delta_z = (self._v_max - self._v_min) / (self.q_func._n_atoms - 1)
        self._z_list = tf.constant(
            [self._v_min + i * self._delta_z for i in range(self.q_func._n_atoms)],
            dtype=tf.float64)
        self.z_list_broadcasted = tf.tile(
            tf.reshape(self._z_list, [1, self.q_func._n_atoms]),
                       tf.constant([self._action_dim, 1]))

    def get_action(self, state, test=False):
        if isinstance(state, LazyFrames):
            state = np.array(state)
        assert isinstance(state, np.ndarray)

        if not test and np.random.rand() < self.epsilon:
            action = np.random.randint(self._action_dim)
        else:
            state = np.expand_dims(state, axis=0).astype(np.float64)
            action_probs = self._get_action_body(tf.constant(state))
            action = tf.argmax(
                tf.reduce_sum(action_probs * self.z_list_broadcasted, axis=2), axis=1)

        return action

    @tf.contrib.eager.defun
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(states, actions, next_states, rewards, done)
                q_func_loss = tf.reduce_mean(tf.square(td_errors) * weights * 0.5)

            q_func_grad = tape.gradient(q_func_loss, self.q_func.trainable_variables)
            self.q_func_optimizer.apply_gradients(zip(q_func_grad, self.q_func.trainable_variables))

            return td_errors, q_func_loss

    @tf.contrib.eager.defun
    def _compute_td_error_body(self, states, actions, next_states, rewards, done):
        actions = tf.cast(actions, dtype=tf.int32)
        with tf.device(self.device):
            indices = tf.concat(
                values=[tf.expand_dims(tf.range(self.batch_size), axis=1),
                        actions], axis=1)
            current_Q = tf.expand_dims(
                tf.gather_nd(self.q_func(states), indices), axis=1)
            print("         current Q: ", self.q_func(states).shape, self.q_func(states).dtype)
            print("reshaped current Q: ", current_Q.shape, current_Q.dtype)
            print("           indices: ", indices.shape, indices.dtype)

            if self._enable_double_dqn:
                raise NotImplementedError
            else:
                rewards = tf.tile(
                    tf.reshape(rewards, [-1, 1]),
                    tf.constant([1, self.q_func._n_atoms]))
                not_done = 1.0 - tf.tile(
                    tf.reshape(done ,[-1, 1]),
                    tf.constant([1, self.q_func._n_atoms]))
                discounts = tf.cast(
                    tf.reshape(self.discount, [-1, 1]), tf.float64)
                target_Q = tf.reshape(
                    self._z_list, [1, self.q_func._n_atoms])
                print(rewards.shape, not_done.shape, discounts.shape, target_Q.shape)
                print(rewards.dtype, not_done.dtype, discounts.dtype, target_Q.dtype)
                target_Q = rewards + not_done * discounts * target_Q
            target_Q = tf.stop_gradient(target_Q)
            td_errors = current_Q - target_Q
        return td_errors
