import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.policies.tfp_gaussian_actor import GaussianActor
from tf2rl.policies.tfp_categorical_actor import CategoricalActor
from tf2rl.envs.atari_wrapper import LazyFrames


class CriticV(tf.keras.Model):
    def __init__(self, state_shape, units, name='critic_v', hidden_activation='relu'):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation=hidden_activation)
        self.l2 = Dense(units[1], name="L2", activation=hidden_activation)
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
    """
    VPG Agent: https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf

    Command Line Args:

        * ``--batch-size`` (int): Batch size of training. The default is ``32``.
        * ``--gpu`` (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        * ``--horizon`` (int): The default is ``2048``.
        * ``--normalize_adv``: Normalize Advantage.
        * ``--enable-gae``: Enable GAE.
    """
    def __init__(
            self,
            state_shape,
            action_dim,
            is_discrete,
            actor=None,
            critic=None,
            actor_critic=None,
            max_action=1.,
            actor_units=(256, 256),
            critic_units=(256, 256),
            lr_actor=1e-3,
            lr_critic=3e-3,
            hidden_activation_actor="relu",
            hidden_activation_critic="relu",
            name="VPG",
            **kwargs):
        """
        Initialize VPG

        Args:
            state_shape (iterable of int):
            action_dim (int):
            is_discrete (bool):
            actor:
            critic:
            actor_critic:
            max_action (float): maximum action size.
            actor_units (iterable of int): Numbers of units at hidden layers of actor. The default is ``(256, 256)``.
            critic_units (iterable of int): Numbers of units at hidden layers of critic. The default is ``(256, 256)``.
            lr_actor (float): Learning rate of actor. The default is ``1e-3``.
            lr_critic (float): Learning rate of critic. The default is ``3e-3``.
            hidden_activation_actor (str): Activation for actor. The default is ``"relu"``.
            hidden_activation_critic (str): Activation for critic. The default is ``"relu"``.
            name (str): Name of agent. The default is ``"VPG"``.
            horizon (int): Number of steps of online episode horizon. The horizon must be multiple of ``batch_size``. The default is ``2048``.
            enable_gae (bool): Enable GAE. The default is ``True``.
            normalize_adv (bool): Normalize Advantage. The default is ``True``.
            entropy_coef (float): Entropy coefficient. The default is ``0.01``.
            vfunc_coef (float): Mixing ratio factor for actor and critic. ``actor_loss + vfunc_coef*critic_loss``
            batch_size (int): Batch size. The default is ``256``.
        """
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
                        hidden_activation=hidden_activation_actor,
                        state_independent_std=True)
            else:
                self.actor = actor
            if critic is None:
                self.critic = CriticV(state_shape, critic_units,
                                      hidden_activation=hidden_activation_critic)
            else:
                self.critic = critic
            self.actor_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_actor)
            self.critic_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_critic)

        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

    def get_action(self, state, test=False):
        """
        Get action and probability

        Args:
            state: Observation state
            test (bool): When ``False`` (default), policy returns exploratory action.

        Returns:
            np.ndarray or float: Selected action
            np.ndarray or float: Log(p)
        """
        if isinstance(state, LazyFrames):
            state = np.array(state)
        msg = "Input instance should be np.ndarray, not {}".format(type(state))
        assert isinstance(state, np.ndarray), msg

        is_single_input = state.ndim == self._state_ndim
        if is_single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
        action, logp = self._get_action_body(state, test)

        if is_single_input:
            return action.numpy()[0], logp.numpy()
        else:
            return action.numpy(), logp.numpy()

    def get_action_and_val(self, state, test=False):
        """
        Get action, probability, and critic value

        Args:
            state: Observation state
            test (bool): When ``False`` (default), policy returns exploratory action.

        Returns:
            np.ndarray: Selected action
            np.ndarray: Log(p)
            np.ndarray: Critic value
        """
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
            action, logp = self.actor(state, test)
            v = self.critic(state)
            return action, logp, v

    @tf.function
    def _get_action_body(self, state, test):
        if self.actor_critic is not None:
            action, logp = self.actor_critic(state, test)
            return action, logp
        else:
            return self.actor(state, test)

    def train(self, states, actions, advantages, logp_olds, returns):
        """
        Train VPG

        Args:
            states
            actions
            advantages
            logp_olds
            returns
        """
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
                critic_loss = tf.reduce_mean(tf.square(td_errors))
            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

        return critic_loss
