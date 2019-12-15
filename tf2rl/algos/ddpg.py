import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss
from tf2rl.networks.network_base import Network, StateActionNetwork


class Actor(Network):
    def __init__(self, state_shape, action_dim, max_action, units=(400, 300), name="Actor"):
        units = np.asarray(units)
        activation = ["relu"] * units.shape[0]
        units = np.append(units,[action_dim],axis=0)
        activation.append("linear")

        super().__init__(input_shape=(1,)+state_shape,
                         units=units,
                         activation=activation,
                         name=name)

        self.max_action = max_action

    def call(self, inputs):
        features = super().call(inputs)
        action = self.max_action * tf.nn.tanh(features)
        return action


class Critic(StateActionNetwork):
    def __init__(self, state_shape, action_dim, units=(400, 300), name="Critic"):
        units = np.asarray(units)
        activation = ["relu"] * units.shape[0]
        units = np.append(units,[1],axis=0)
        activation.append("linear")

        input_shape = np.asarray((1,) + state_shape)
        input_shape[1] += action_dim

        super().__init__(input_shape=input_shape,
                         units=units,
                         activation=activation,
                         name=name)


class DDPG(OffPolicyAgent):
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
            action += tf.random.normal(shape=action.shape,
                                       mean=0., stddev=sigma, dtype=tf.float32)
            return tf.clip_by_value(action, -max_action, max_action)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        actor_loss, critic_loss, td_errors = self._train_body(
            states, actions, next_states, rewards, done, weights)

        if actor_loss is not None:
            tf.summary.scalar(name=self.policy_name+"/actor_loss",
                              data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/critic_loss",
                          data=critic_loss)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(
                    states, actions, next_states, rewards, done)
                critic_loss = tf.reduce_mean(
                    huber_loss(td_errors, delta=self.max_grad) * weights)

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                next_action = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic([states, next_action]))

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
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        td_errors = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.abs(np.ravel(td_errors.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. - dones
            target_Q = self.critic_target(
                [next_states, self.actor_target(next_states)])
            target_Q = rewards + (not_dones * self.discount * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            current_Q = self.critic([states, actions])
            td_errors = target_Q - current_Q
        return td_errors
