import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.networks.noisy_dense import NoisyDense
from tf2rl.envs.atari_wrapper import LazyFrames
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss


class QFunc(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(32, 32),
                 name="CategoricalQFunc", enable_dueling_dqn=False,
                 enable_noisy_dqn=False, n_atoms=51):
        super().__init__(name=name)
        self._enable_dueling_dqn = enable_dueling_dqn
        self._enable_noisy_dqn = enable_noisy_dqn

        self._action_dim = action_dim
        self._n_atoms = n_atoms
        action_dim = (action_dim + int(enable_dueling_dqn)) * n_atoms

        DenseLayer = NoisyDense if enable_noisy_dqn else Dense

        self.l1 = DenseLayer(units[0], name="L1", activation="relu")
        self.l2 = DenseLayer(units[1], name="L2", activation="relu")
        self.l3 = DenseLayer(action_dim, name="L3", activation="linear")

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        features = self.l3(features)
        if self._enable_dueling_dqn:
            features = tf.reshape(
                features, (-1, self._action_dim + 1, self._n_atoms))  # [batch_size, action_dim, n_atoms]
            v_values = tf.reshape(
                features[:, 0], (-1, 1, self._n_atoms))
            advantages = tf.reshape(
                features[:, 1:], [-1, self._action_dim, self._n_atoms])
            features = v_values + (advantages - tf.expand_dims(
                tf.reduce_mean(advantages, axis=1), axis=1))
        else:
            features = tf.reshape(
                features, (-1, self._action_dim, self._n_atoms))  # [batch_size, action_dim, n_atoms]
        # [batch_size, action_dim, n_atoms]
        q_dist = tf.keras.activations.softmax(features, axis=2)
        return tf.clip_by_value(q_dist, 1e-8, 1.0 - 1e-8)

    @property
    def n_atoms(self):
        return self._n_atoms


class CategoricalDQN(OffPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            q_func=None,
            name="DQN",
            lr=0.001,
            adam_eps=1e-07,
            units=(32, 32),
            epsilon=0.1,
            epsilon_min=None,
            epsilon_decay_step=int(1e6),
            n_warmup=int(1e4),
            target_replace_interval=int(5e3),
            memory_capacity=int(1e6),
            enable_double_dqn=False,
            enable_dueling_dqn=False,
            enable_noisy_dqn=False,
            **kwargs):
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        q_func = q_func if q_func is not None else QFunc
        # Define and initialize Q-function network
        kwargs_dqn = {
            "state_shape": state_shape,
            "action_dim": action_dim,
            "units": units,
            "enable_dueling_dqn": enable_dueling_dqn,
            "enable_noisy_dqn": enable_noisy_dqn}
        self.q_func = q_func(**kwargs_dqn)
        self.q_func_target = q_func(**kwargs_dqn)
        self.q_func_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=adam_eps)
        update_target_variables(self.q_func_target.weights,
                                self.q_func.weights, tau=1.)

        self._action_dim = action_dim
        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

        # Distributional DQN
        self._v_max, self._v_min = 10., -10.
        self._delta_z = (self._v_max - self._v_min) / (self.q_func._n_atoms - 1)
        self._z_list = tf.constant(
            [self._v_min + i *
             self._delta_z for i in range(self.q_func._n_atoms)],
            dtype=tf.float32)
        self._z_list_broadcasted = tf.tile(
            tf.reshape(self._z_list, [1, self.q_func._n_atoms]),
            tf.constant([self._action_dim, 1]))

        # Set hyper-parameters
        if epsilon_min is not None and not enable_noisy_dqn:
            assert epsilon > epsilon_min
            self.epsilon_min = epsilon_min
            self.epsilon_decay_rate = (epsilon - epsilon_min) / epsilon_decay_step
            self.epsilon = max(epsilon - self.epsilon_decay_rate * self.n_warmup,
                               self.epsilon_min)
        else:
            epsilon = epsilon if not enable_noisy_dqn else 0.
            self.epsilon = epsilon
            self.epsilon_min = epsilon
            self.epsilon_decay_rate = 0.
        self.target_replace_interval = target_replace_interval
        self.n_update = 0

        # DQN variants
        self._enable_double_dqn = enable_double_dqn
        self._enable_noisy_dqn = enable_noisy_dqn

    def get_action(self, state, test=False, tensor=False):
        if isinstance(state, LazyFrames):
            state = np.array(state)
        if not tensor:
            assert isinstance(state, np.ndarray)
        is_single_input = state.ndim == self._state_ndim

        if not test and np.random.rand() < self.epsilon:
            if is_single_input:
                action = np.random.randint(self._action_dim)
            else:
                action = np.array([np.random.randint(self._action_dim)
                                   for _ in range(state.shape[0])], dtype=np.int64)
            if tensor:
                return tf.convert_to_tensor(action)
            else:
                return action

        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_input else state
        action = self._get_action_body(tf.constant(state))
        if tensor:
            return action
        else:
            if is_single_input:
                return action.numpy()[0]
            else:
                return action.numpy()

    @tf.function
    def _get_action_body(self, state):
        action_probs = self.q_func(state)
        return tf.argmax(
            tf.reduce_sum(action_probs * self._z_list_broadcasted, axis=2),
            axis=1)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        td_errors, q_func_loss = self._train_body(
            states, actions, next_states, rewards, done, weights)

        tf.summary.scalar(name=self.policy_name + "/q_func_Loss", data=q_func_loss)

        # TODO: Remove following by using tf.global_step
        self.n_update += 1
        # Update target networks
        if self.n_update % self.target_replace_interval == 0:
            update_target_variables(
                self.q_func_target.weights, self.q_func.weights, tau=1.)

        # Update exploration rate
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate * self.update_interval,
                           self.epsilon_min)
        tf.summary.scalar(name=self.policy_name + "/epsilon", data=self.epsilon)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(
                    states, actions, next_states, rewards, done)
                q_func_loss = tf.reduce_mean(
                    huber_loss(tf.negative(td_errors),
                               delta=self.max_grad) * weights)

            q_func_grad = tape.gradient(
                q_func_loss, self.q_func.trainable_variables)
            self.q_func_optimizer.apply_gradients(
                zip(q_func_grad, self.q_func.trainable_variables))

            return td_errors, q_func_loss

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        # TODO: fix this ugly conversion
        if isinstance(actions, tf.Tensor):
            actions = tf.expand_dims(actions, axis=1)
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)

        return self._compute_td_error_body(
            states, actions, next_states, rewards, dones)

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        """
        Args:
            states:
            actions:
            next_states:
            rewards:
                Shape should be (batch_size, 1)
            dones:
                Shape should be (batch_size, 1)

        Returns:

        """
        batch_size = states.shape[0]
        actions = tf.cast(actions, dtype=tf.int32)

        # tf.print(tf.expand_dims(tf.range(batch_size), axis=1).shape,
        #          actions.shape,
        #          tf.concat(
        #              values=(tf.expand_dims(tf.range(batch_size), axis=1),
        #                      actions), axis=1).shape
        #          )

        with tf.device(self.device):
            rewards = tf.tile(
                tf.reshape(rewards, [-1, 1]),
                tf.constant([1, self.q_func.n_atoms]))  # [batch_size, n_atoms]
            not_done = 1.0 - tf.tile(
                tf.reshape(dones, [-1, 1]),
                tf.constant([1, self.q_func.n_atoms]))  # [batch_size, n_atoms]
            discounts = tf.cast(
                tf.reshape(self.discount, [-1, 1]), tf.float32)  # [1, 1]
            z = tf.reshape(
                self._z_list, [1, self.q_func.n_atoms])  # [1, n_atoms]
            z = rewards + not_done * discounts * z  # [batch_size, n_atoms]
            z = tf.clip_by_value(z, self._v_min, self._v_max)  # [batch_size, n_atoms]
            b = (z - self._v_min) / self._delta_z  # [batch_size, n_atoms]

            index_help = tf.expand_dims(
                tf.tile(
                    tf.reshape(tf.range(batch_size), [-1, 1]),
                    tf.constant([1, self.q_func.n_atoms])),
                -1)  # [batch_size, n_atoms, 1]
            u, l = tf.math.ceil(b), tf.math.floor(b)  # [batch_size, n_atoms]
            u_id = tf.concat(
                [index_help, tf.expand_dims(tf.cast(u, tf.int32), -1)],
                axis=2)  # [batch_size, n_atoms, n_action]
            l_id = tf.concat(
                [index_help, tf.expand_dims(tf.cast(l, tf.int32), -1)],
                axis=2)  # [batch_size, n_atoms, n_action]

            target_q_next_dist = self.q_func_target(
                next_states)  # [batch_size, n_action, n_atoms]

            if self._enable_double_dqn:
                # TODO: Check this implementation is correct
                target_q_next_dist = tf.gather_nd(
                    target_q_next_dist,
                    tf.concat(
                        [tf.reshape(tf.range(batch_size), [-1, 1]),
                         tf.reshape(actions, [-1, 1])],
                        axis=1))
            else:
                target_q_next_sum = tf.reduce_sum(
                    target_q_next_dist * self._z_list_broadcasted, axis=2)  # [batch_size, n_action]
                actions_by_target_q = tf.cast(
                    tf.argmax(target_q_next_sum, axis=1),
                    tf.int32)  # [batch_size,]
                target_q_next_dist = tf.gather_nd(
                    target_q_next_dist,
                    tf.concat(
                        [tf.reshape(tf.range(batch_size), [-1, 1]),
                         tf.reshape(actions_by_target_q, [-1, 1])],
                        axis=1))  # [batch_size, n_atoms]

            action_indices = tf.concat(
                values=(tf.expand_dims(tf.range(batch_size), axis=1),
                        actions), axis=1)
            current_q_dist = tf.gather_nd(
                self.q_func(states), action_indices)  # [batch_size, n_atoms]

            td_errors = tf.reduce_sum(
                target_q_next_dist * (u - b) * tf.math.log(tf.gather_nd(current_q_dist, l_id)) +
                target_q_next_dist * (b - l) * tf.math.log(tf.gather_nd(current_q_dist, u_id)),
                axis=1)

        return td_errors

    @staticmethod
    def get_argument(parser=None):
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--enable-double-dqn', action='store_true')
        parser.add_argument('--enable-dueling-dqn', action='store_true')
        parser.add_argument('--enable-noisy-dqn', action='store_true')
        return parser
