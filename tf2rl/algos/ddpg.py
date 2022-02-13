import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables


class Actor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action, units=(400, 300), name="Actor"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(action_dim, name="L3")

        self.max_action = max_action

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        action = self.max_action * tf.nn.tanh(features)
        return action


class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(400, 300), name="Critic"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(1, name="L3")

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        with tf.device("/cpu:0"):
            self(dummy_state, dummy_action)

    def call(self, states, actions):
        features = tf.concat((states, actions), axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        values = self.l3(features)
        return tf.squeeze(values, axis=1)


class DDPG(OffPolicyAgent):
    """
    DDPG agent: https://arxiv.org/abs/1509.02971

    Command Line Args:

        * ``--n-warmup`` (int): Number of warmup steps before training. The default is ``1e4``.
        * ``--batch-size`` (int): Batch size for training. The default is ``32``.
        * ``--gpu`` (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        * ``--memory-capacity`` (int): Replay Buffer size. The default is ``1e6``.
    """
    def __init__(
            self,
            state_shape,
            action_dim,
            name="DDPG",
            max_action=1.,
            lr_actor=0.001,
            lr_critic=0.001,
            actor_units=(400, 300),
            critic_units=(400, 300),
            sigma=0.1,
            tau=0.005,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            **kwargs):
        """
        Initialize DDPG agent

        Args:
            state_shape (iterable of int):
            action_dim (int):
            name (str): Name of agent. The default is ``"DDPG"``.
            max_action (float): Size of maximum action. (``-max_action`` <= action <= ``max_action``). The degault is ``1``.
            lr_actor (float): Learning rate for actor network. The default is ``0.001``.
            lr_critic (float): Learning rage for critic network. The default is ``0.001``.
            actor_units (iterable of int): Number of units at hidden layers of actor.
            critic_units (iterable of int): Number of units at hidden layers of critic.
            sigma (float): Standard deviation of Gaussian noise. The default is ``0.1``.
            tau (float): Weight update ratio for target network. ``target = (1-tau)*target + tau*network`` The default is ``0.005``.
            n_warmup (int): Number of warmup steps before training. The default is ``1e4``.
            memory_capacity (int): Replay Buffer size. The default is ``1e4``.
            batch_size (int): Batch size. The default is ``256``.
            discount (float): Discount factor. The default is ``0.99``.
            max_grad (float): Maximum gradient. The default is ``10``.
            gpu (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        """
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        # Define and initialize Actor network
        self.actor = Actor(state_shape, action_dim, max_action, actor_units)
        self.actor_target = Actor(
            state_shape, action_dim, max_action, actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        update_target_variables(self.actor_target.weights,
                                self.actor.weights, tau=1.)

        # Define and initialize Critic network
        self.critic = Critic(state_shape, action_dim, critic_units)
        self.critic_target = Critic(state_shape, action_dim, critic_units)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_critic)
        update_target_variables(
            self.critic_target.weights, self.critic.weights, tau=1.)

        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau

    def get_action(self, state, test=False, tensor=False):
        """
        Get action

        Args:
            state: Observation state
            test (bool): When ``False`` (default), policy returns exploratory action.
            tensor (bool): When ``True``, return type is ``tf.Tensor``

        Returns:
            tf.Tensor or np.ndarray or float: Selected action
        """
        is_single_state = len(state.shape) == 1
        if not tensor:
            assert isinstance(state, np.ndarray)
        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(
            tf.constant(state), self.sigma * (1. - test),
            tf.constant(self.actor.max_action, dtype=tf.float32))
        if tensor:
            return action
        else:
            return action.numpy()[0] if is_single_state else action.numpy()

    @tf.function
    def _get_action_body(self, state, sigma, max_action):
        with tf.device(self.device):
            action = self.actor(state)
            if sigma > 0.:
                action += tf.random.normal(shape=action.shape,
                                           mean=0., stddev=sigma, dtype=tf.float32)
            return tf.clip_by_value(action, -max_action, max_action)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        """
        Train DDPG

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
        actor_loss, critic_loss, td_errors = self._train_body(
            states, actions, next_states, rewards, done, weights)

        if actor_loss is not None:
            tf.summary.scalar(name=self.policy_name + "/actor_loss",
                              data=actor_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_loss",
                          data=critic_loss)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(
                    states, actions, next_states, rewards, dones)
                critic_loss = tf.reduce_mean(td_errors ** 2)

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                sample_actions = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic(states, sample_actions))

            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            update_target_variables(
                self.critic_target.weights, self.critic.weights, self.tau)
            update_target_variables(
                self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, td_errors

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
            tf.Tensor: TD error
        """
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        td_errors = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.abs(np.ravel(td_errors.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        assert len(dones.shape) == 2
        assert len(rewards.shape) == 2
        rewards = tf.squeeze(rewards, axis=1)
        dones = tf.squeeze(dones, axis=1)

        with tf.device(self.device):
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)
            next_act_target = self.actor_target(next_states)
            next_q_target = self.critic_target(next_states, next_act_target)
            target_q = rewards + not_dones * self.discount * next_q_target
            current_q = self.critic(states, actions)
            td_errors = tf.stop_gradient(target_q) - current_q
        return td_errors
