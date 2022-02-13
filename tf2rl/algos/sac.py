import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.policies.tfp_gaussian_actor import GaussianActor


class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, critic_units=(256, 256), name='qf'):
        super().__init__(name=name)

        self.base_layers = []
        for unit in critic_units:
            self.base_layers.append(Dense(unit, activation='relu'))
        self.out_layer = Dense(1, name="Q", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)

    def call(self, states, actions):
        features = tf.concat((states, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)


class SAC(OffPolicyAgent):
    """
    Soft Actor-Critic (SAC) Agent: https://arxiv.org/abs/1801.01290

    Command Line Args:

        * ``--n-warmup`` (int): Number of warmup steps before training. The default is ``1e4``.
        * ``--batch-size`` (int): Batch size of training. The default is ``32``.
        * ``--gpu`` (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        * ``--memory-capacity`` (int): Replay Buffer size. The default is ``1e6``.
        * ``--alpha`` (float): Temperature parameter. The default is ``0.2``.
        * ``--auto-alpha``: Automatic alpha tuning.
    """
    def __init__(
            self,
            state_shape,
            action_dim,
            name="SAC",
            max_action=1.,
            lr=3e-4,
            lr_alpha=3e-4,
            actor_units=(256, 256),
            critic_units=(256, 256),
            tau=5e-3,
            alpha=.2,
            auto_alpha=False,
            init_temperature=None,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            **kwargs):
        """
        Initialize SAC

        Args:
            state_shape (iterable of int):
            action_dim (int):
            name (str): Name of network. The default is ``"SAC"``
            max_action (float):
            lr (float): Learning rate. The default is ``3e-4``.
            lr_alpha (alpha): Learning rate for alpha. The default is ``3e-4``.
            actor_units (iterable of int): Numbers of units at hidden layers of actor. The default is ``(256, 256)``.
            critic_units (iterable of int): Numbers of units at hidden layers of critic. The default is ``(256, 256)``.
            tau (float): Target network update rate.
            alpha (float): Temperature parameter. The default is ``0.2``.
            auto_alpha (bool): Automatic alpha tuning.
            init_temperature (float): Initial temperature
            n_warmup (int): Number of warmup steps before training. The default is ``int(1e4)``.
            memory_capacity (int): Replay Buffer size. The default is ``int(1e6)``.
            batch_size (int): Batch size. The default is ``256``.
            discount (float): Discount factor. The default is ``0.99``.
            max_grad (float): Maximum gradient. The default is ``10``.
            gpu (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        """
        super().__init__(
            name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        self._setup_actor(state_shape, action_dim, actor_units, lr, max_action)
        self._setup_critic_q(state_shape, action_dim, critic_units, lr)

        # Set hyper-parameters
        self.tau = tau
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if init_temperature is not None:
                init_log_alpha = np.log(init_temperature)
                self.log_alpha = tf.Variable(init_log_alpha, dtype=tf.float32)
            else:
                self.log_alpha = tf.Variable(0., dtype=tf.float32)
            self.alpha = tfp.util.DeferredTensor(pretransformed_input=self.log_alpha, transform_fn=tf.exp)
            self.target_alpha = -action_dim
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_alpha, beta_1=0.5)
        else:
            self.alpha = alpha

        self.state_ndim = len(state_shape)

    def _setup_actor(self, state_shape, action_dim, actor_units, lr, max_action=1.):
        self.actor = GaussianActor(
            state_shape, action_dim, max_action, squash=True, units=actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_q(self, state_shape, action_dim, critic_units, lr):
        self.qf1 = CriticQ(state_shape, action_dim, critic_units, name="qf1")
        self.qf2 = CriticQ(state_shape, action_dim, critic_units, name="qf2")
        self.qf1_target = CriticQ(state_shape, action_dim, critic_units, name="qf1_target")
        self.qf2_target = CriticQ(state_shape, action_dim, critic_units, name="qf2_target")
        update_target_variables(self.qf1_target.weights, self.qf1.weights, tau=1.)
        update_target_variables(self.qf2_target.weights, self.qf2.weights, tau=1.)
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def get_action(self, state, test=False):
        """
        Get action

        Args:
            state: Observation state
            test (bool): When ``False`` (default), policy returns exploratory action.

        Returns:
            tf.Tensor or float: Selected action
        """
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = np.expand_dims(state, axis=0) if is_single_state else state
        action = self._get_action_body(tf.constant(state), test)

        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state, test):
        actions, log_pis = self.actor(state, test)
        return actions

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        """
        Train SAC

        Args:
            states
            actions
            next_states
            rewards
            done
            weights (optional): Weights for importance sampling
        """
        if weights is None:
            weights = np.ones_like(rewards)

        td_errors, qf_loss = self._update_critic(
            states, actions, next_states, rewards, dones, weights)
        tf.summary.scalar(name=self.policy_name + "/critic_loss", data=qf_loss)

        actor_loss, logp_min, logp_max, logp_mean, alpha_loss = self._update_actor(states)
        tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
        tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
        tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
        if self.auto_alpha:
            tf.summary.scalar(name=self.policy_name + "/log_alpha", data=self.log_alpha)
            tf.summary.scalar(name=self.policy_name + "/logp_mean+target", data=logp_mean + self.target_alpha)
        tf.summary.scalar(name=self.policy_name + "/alpha", data=self.alpha)
        tf.summary.scalar(name=self.policy_name + "/alpha_loss", data=alpha_loss)

        return td_errors

    @tf.function
    def _update_critic(self, states, actions, next_states, rewards, dones, weights):
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2
            rewards = tf.squeeze(rewards, axis=1)
            dones = tf.squeeze(dones, axis=1)

            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                next_actions, next_logps = self.actor(next_states)
                next_target_q1 = tf.stop_gradient(self.qf1_target(next_states, next_actions))
                next_target_q2 = tf.stop_gradient(self.qf2_target(next_states, next_actions))
                min_next_target_q = tf.minimum(next_target_q1, next_target_q2)

                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * (min_next_target_q - self.alpha * next_logps))

                current_q1 = self.qf1(states, actions)
                current_q2 = self.qf2(states, actions)
                td_loss_q1 = tf.reduce_mean((target_q - current_q1) ** 2)
                td_loss_q2 = tf.reduce_mean((target_q - current_q2) ** 2)  # Eq.(6)

            q1_grad = tape.gradient(td_loss_q1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss_q2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))
            update_target_variables(self.qf1_target.weights, self.qf1.weights, self.tau)
            update_target_variables(self.qf2_target.weights, self.qf2.weights, self.tau)

        return td_loss_q1 + td_loss_q2, td_loss_q1

    @tf.function
    def _update_actor(self, states):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                sample_actions, logp = self.actor(states)  # Resample actions to update V
                current_q1 = self.qf1(states, sample_actions)
                current_q2 = self.qf2(states, sample_actions)
                current_min_q = tf.minimum(current_q1, current_q2)

                # Compute loss of policy
                policy_loss = tf.reduce_mean(self.alpha * logp - current_min_q)  # Eq.(12)

                # Compute loss of temperature parameter for entropy
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean(
                        (self.alpha * tf.stop_gradient(logp + self.target_alpha)))
                else:
                    alpha_loss = 0.

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(
                    zip(alpha_grad, [self.log_alpha]))

        return policy_loss, tf.reduce_min(logp), tf.reduce_max(logp), tf.reduce_mean(logp), alpha_loss

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        """
        Compute TD error

        Args:
            states
            actions
            next_states
            rewars
            dones

        Returns
            np.ndarray: TD error
        """
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
            next_actions, next_logps = self.actor(next_states)
            next_target_q1 = tf.stop_gradient(self.qf1_target(next_states, next_actions))
            next_target_q2 = tf.stop_gradient(self.qf2_target(next_states, next_actions))
            min_next_target_q = tf.minimum(next_target_q1, next_target_q2)

            target_q = tf.stop_gradient(
                rewards + not_dones * self.discount * (min_next_target_q - self.alpha * next_logps))

            current_q1 = self.qf1(states, actions)
            td_errors_q1 = target_q - current_q1

        return td_errors_q1

    @staticmethod
    def get_argument(parser=None):
        """
        Create or update argument parser for command line program

        Args:
            parser (argparse.ArgParser, optional): argument parser

        Returns:
            argparse.ArgParser: argument parser
        """
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--auto-alpha', action="store_true")
        return parser
