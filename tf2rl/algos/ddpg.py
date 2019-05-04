import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import OffPolicyAgent
import tf2rl.misc.target_update_ops as target_update


class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action, units=[400, 300], name="Actor"):
        super().__init__(name=name)

        self.l1 = tf.keras.layers.Dense(units[0], name="L1")
        self.l2 = tf.keras.layers.Dense(units[1], name="L2")
        self.l3 = tf.keras.layers.Dense(action_dim, name="L3")

        self.max_action = max_action

        self(tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float64)), device="/cpu:0")

    def call(self, inputs, device="/gpu:0"):
        with tf.device(device):
            features = tf.nn.relu(self.l1(inputs))
            features = tf.nn.relu(self.l2(features))
            features = self.l3(features)
            action = self.max_action * tf.nn.tanh(features)
        return action


class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, units=[400, 300], name="Critic"):
        super().__init__(name=name)

        self.l1 = tf.keras.layers.Dense(units[0], name="L1")
        self.l2 = tf.keras.layers.Dense(units[1], name="L2")
        self.l3 = tf.keras.layers.Dense(1, name="L3")

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float64))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float64))
        self([dummy_state, dummy_action], device="/cpu:0")

    def call(self, inputs, device="/gpu:0"):
        with tf.device(device):
            states, actions = inputs
            features = tf.concat([states, actions], axis=1)
            features = tf.nn.relu(self.l1(features))
            features = tf.nn.relu(self.l2(features))
            features = self.l3(features)
        return features


class DDPG(OffPolicyAgent):
    def __init__(
            self,
            state_dim,
            action_dim,
            name="DDPG",
            max_action=1.,
            lr_actor=0.001,
            lr_critic=0.001,
            actor_units=[400, 300],
            critic_units=[400, 300],
            sigma=0.1,
            tau=0.005,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            **kwargs):
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        # Define and initialize Actor network
        self.actor = Actor(state_dim, action_dim, max_action, actor_units)
        self.actor_target = Actor(state_dim, action_dim, max_action, actor_units)
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=lr_actor)
        for param, target_param in zip(self.actor.weights, self.actor_target.weights):
            target_param.assign(param)

        # Define and initialize Critic network
        self.critic = Critic(state_dim, action_dim, critic_units)
        self.critic_target = Critic(state_dim, action_dim, critic_units)
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=lr_critic)
        for param, target_param in zip(self.critic.weights, self.critic_target.weights):
            target_param.assign(param)

        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)
        assert len(state.shape) == 1

        state = np.expand_dims(state, axis=0).astype(np.float64)
        action = self._get_action_body(tf.constant(state))
        action = action.numpy()[0]
        if not test:
            action += np.random.normal(0, self.sigma, size=action.shape)

        return action.clip(-self.actor.max_action, self.actor.max_action)

    @tf.contrib.eager.defun
    def _get_action_body(self, state):
        return self.actor(state, device=self.device)

    def train(self, replay_buffer):
        samples = replay_buffer.sample(self.batch_size)
        state, next_state, action, reward, done = samples["obs"], samples["next_obs"], samples["act"], samples["rew"], samples["done"]
        done = np.array(done, dtype=np.float64)
        actor_loss, critic_loss = self._train_body(state, action, next_state, reward, done)

        tf.contrib.summary.scalar(name="ActorLoss", tensor=actor_loss, family="loss")
        tf.contrib.summary.scalar(name="CriticLoss", tensor=critic_loss, family="loss")

        return actor_loss, critic_loss

    @tf.contrib.eager.defun
    def _train_body(self, states, actions, next_states, rewards, done):
        with tf.device(self.device):
            not_done = 1. - done

            with tf.GradientTape() as tape:
                target_Q = self.critic_target([next_states, self.actor_target(next_states)],
                                              device=self.device)
                target_Q = rewards + (not_done * self.discount * target_Q)
                target_Q = tf.stop_gradient(target_Q)
                current_Q = self.critic([states, actions], device=self.device)
                critic_loss = tf.reduce_mean(tf.keras.losses.MSE(current_Q, target_Q))

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                next_action = self.actor(states, device=self.device)
                actor_loss = -tf.reduce_mean(self.critic([states, next_action], device=self.device))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            target_update.update_target_variables(self.critic_target.weights, self.critic.weights, self.tau)
            target_update.update_target_variables(self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss

    def compute_td_error(self, states, actions, next_states, rewards, done):
        td_errors, critic_loss = self._compute_td_error_body(states, actions, next_states, rewards, done)
        print(critic_loss.numpy())
        return td_errors.numpy()

    @tf.contrib.eager.defun
    def _compute_td_error_body(self, states, actions, next_states, rewards, done):
        with tf.device(self.device):
            not_done = 1. - done
            target_Q = self.critic_target(
                [next_states, self.actor_target(next_states)], device=self.device)
            target_Q = rewards + (not_done * self.discount * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            current_Q = self.critic([states, actions], device=self.device)
            td_errors = target_Q - current_Q
        return td_errors, tf.reduce_mean(tf.keras.losses.MSE(current_Q, target_Q))
