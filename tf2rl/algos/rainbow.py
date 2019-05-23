import numpy as np
import tensorflow as tf

from tf2rl.algos.dqn import DQN
from tf2rl.envs.atari_wrapper import LazyFrames


class Rainbow(DQN):
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

    def get_action(self, state, test=False):
        if isinstance(state, LazyFrames):
            state = np.array(state)
        assert isinstance(state, np.ndarray)

        if not test and np.random.rand() < self.epsilon:
            action = np.random.randint(self._action_dim)
        else:
            state = np.expand_dims(state, axis=0).astype(np.float32)
            action_probs = self._get_action_body(tf.constant(state))
            action = tf.argmax(
                tf.reduce_sum(action_probs * self._z_list_broadcasted, axis=2),
                axis=1)
            action = action.numpy()[0]

        return action

    @tf.contrib.eager.defun
    def _compute_td_error_body(self, states, actions, next_states, rewards, done):
        actions = tf.cast(actions, dtype=tf.int32)
        with tf.device(self.device):
            rewards = tf.tile(
                tf.reshape(rewards, [-1, 1]),
                tf.constant([1, self.q_func._n_atoms]))  # [batch_size, n_atoms]
            not_done = 1.0 - tf.tile(
                tf.reshape(done ,[-1, 1]),
                tf.constant([1, self.q_func._n_atoms]))  # [batch_size, n_atoms]
            discounts = tf.cast(
                tf.reshape(self.discount, [-1, 1]), tf.float32)
            z = tf.reshape(
                self._z_list, [1, self.q_func._n_atoms])  # [1, n_atoms]
            z = rewards + not_done * discounts * z  # [batch_size, n_atoms]
            z = tf.clip_by_value(z, self._v_min, self._v_max)  # [batch_size, n_atoms]
            b = (z - self._v_min) / self._delta_z  # [batch_size, n_atoms]

            index_help = tf.expand_dims(
                tf.tile(
                    tf.reshape(tf.range(self.batch_size), [-1, 1]),
                    tf.constant([1, self.q_func._n_atoms])),
                -1)  # [batch_size, n_atoms, 1]
            u, l = tf.ceil(b), tf.floor(b)  # [batch_size, n_atoms]
            u_id = tf.concat(
                [index_help, tf.expand_dims(tf.cast(u, tf.int32), -1)],
                axis=2)  # [batch_size, n_atoms, 2]
            l_id = tf.concat(
                [index_help, tf.expand_dims(tf.cast(l, tf.int32), -1)],
                axis=2)  # [batch_size, n_atoms, 2]

            target_Q_next_dist = self.q_func_target(next_states)  # [batch_size, n_action, n_atoms]
            target_Q_next_dist = tf.gather_nd(
                target_Q_next_dist,
                tf.concat(
                    [tf.reshape(tf.range(self.batch_size), [-1, 1]),
                        tf.reshape(actions, [-1, 1])],
                    axis=1))

            action_indices = tf.concat(
                values=[tf.expand_dims(tf.range(self.batch_size), axis=1),
                        actions], axis=1)
            current_Q_dist = tf.gather_nd(
                self.q_func(states), action_indices)  # [batch_size, n_atoms]

            td_errors = tf.reduce_sum(
                target_Q_next_dist * (u - b) * tf.log(
                    tf.gather_nd(current_Q_dist, l_id)) + \
                target_Q_next_dist * (b - l) * tf.log(
                    tf.gather_nd(current_Q_dist, u_id)),
                axis=1)

        return td_errors
