import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.policies.gaussian_actor import GaussianActor
from tf2rl.policies.categorical_actor import CategoricalActor
from tf2rl.envs.atari_wrapper import LazyFrames


class CriticV(tf.keras.Model):
    def __init__(self, state_shape, units, name='qf'):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation='relu')
        self.l2 = Dense(units[1], name="L2", activation='relu')
        self.l3 = Dense(1, name="L2", activation='linear')

        with tf.device('/cpu:0'):
            self(tf.constant(
                np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)


class VPG(OnPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            is_discrete,
            actor=None,
            critic=None,
            actor_critic=None,
            max_action=1.,
            actor_units=[256, 256],
            critic_units=[256, 256],
            lr_actor=1e-3,
            lr_critic=3e-3,
            tanh_std=False,
            fix_std=False,
            const_std=0.3,
            hidden_activation="relu",
            name="VPG",
            **kwargs):
        super().__init__(name=name, **kwargs)
        self._is_discrete = is_discrete

        # TODO: clean codes
        if actor_critic is not None:
            self.actor_critic = actor_critic
            self.actor_critic_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_actor)
            self.actor = None
            self.critic = None
        else:
            self.actor_critic = None
            if actor is None:
                if is_discrete:
                    self.actor = CategoricalActor(
                        state_shape, action_dim, actor_units)
                else:
                    self.actor = GaussianActor(
                        state_shape, action_dim, max_action, actor_units,
                        hidden_activation=hidden_activation,
                        fix_std=fix_std, tanh_std=tanh_std,
                        const_std=const_std, state_independent_std=True)
            else:
                self.actor = actor
            if critic is None:
                self.critic = CriticV(state_shape, critic_units)
            else:
                self.critic = critic
            self.actor_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_actor)
            self.critic_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_critic)

        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

    def get_action(self, state, test=False):
        if isinstance(state, LazyFrames):
            state = np.array(state)
        assert isinstance(state, np.ndarray), \
            "Input instance should be np.ndarray, not {}".format(type(state))

        is_single_input = state.ndim == self._state_ndim
        if is_single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
        action, logp, _ = self._get_action_body(state, test)

        if is_single_input:
            return action.numpy()[0], logp.numpy()
        else:
            return action.numpy(), logp.numpy()

    def get_action_and_val(self, state, test=False):
        if isinstance(state, LazyFrames):
            state = np.array(state)
        is_single_input = state.ndim == self._state_ndim
        if is_single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)

        action, logp, v = self._get_action_logp_v_body(state, test)

        if is_single_input:
            v = v[0]
            action = action[0]

        return action.numpy(), logp.numpy(), v.numpy()

    @tf.function
    def _get_action_logp_v_body(self, state, test):
        if self.actor_critic:
            return self.actor_critic(state, test)
        else:
            action, logp, _ = self.actor(state, test)
            v = self.critic(state)
            return action, logp, v

    @tf.function
    def _get_action_body(self, state, test):
        if self.actor_critic is not None:
            action, logp, param = self.actor_critic(state, test)
            return action, logp, param
        else:
            return self.actor(state, test)

    def train(self, states, actions, advantages, logp_olds, returns):
        # Train actor and critic
        actor_loss, logp_news = self._train_actor_body(
            states, actions, advantages, logp_olds)
        critic_loss = self._train_critic_body(states, returns)
        # Visualize results in TensorBoard
        tf.summary.scalar(name=self.policy_name+"/actor_loss",
                          data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/logp_max",
                          data=np.max(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_min",
                          data=np.min(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_mean",
                          data=np.mean(logp_news))
        tf.summary.scalar(name=self.policy_name+"/adv_max",
                          data=np.max(advantages))
        tf.summary.scalar(name=self.policy_name+"/adv_min",
                          data=np.min(advantages))
        tf.summary.scalar(name=self.policy_name+"/kl",
                          data=tf.reduce_mean(logp_olds - logp_news))
        tf.summary.scalar(name=self.policy_name +
                          "/critic_loss", data=critic_loss)
        return actor_loss, critic_loss

    @tf.function
    def _train_actor_body(self, states, actions, advantages, logp_olds):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                log_probs = self.actor.compute_log_probs(states, actions)
                weights = tf.stop_gradient(tf.squeeze(advantages))
                # + lambda * entropy
                actor_loss = tf.reduce_mean(-log_probs * weights)
            actor_grads = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grads, self.actor.trainable_variables))

        return actor_loss, log_probs

    @tf.function
    def _train_critic_body(self, states, returns):
        with tf.device(self.device):
            # Train baseline
            with tf.GradientTape() as tape:
                current_V = self.critic(states)
                td_errors = tf.squeeze(returns) - current_V
                critic_loss = tf.reduce_mean(0.5 * tf.square(td_errors))
            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

        return critic_loss
