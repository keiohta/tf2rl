import numpy as np
import tensorflow as tf

from tf2rl.algos.ddpg import DDPG
from tf2rl.misc.target_update_ops import update_target_variables


class BiResDDPG(DDPG):
    """
    Bi-Res-DDPG Agent: https://arxiv.org/abs/1905.01072

    Command Line Args:

        * ``--n-warmup`` (int): Number of warmup steps before training. The default is ``1e4``.
        * ``--batch-size`` (int): Batch size for training. The default is ``32``.
        * ``--gpu`` (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        * ``--memory-capacity`` (int): Replay Buffer size. The default is ``1e6``.
        * ``--eta`` (float): Gradient mixing factor. The default is ``0.05``.
    """

    def __init__(self, eta=0.05, name="BiResDDPG", **kwargs):
        """
        Initialize BiResDDPG agent

        Args:
            eta (float): Gradients mixing factor.
            name (str): Name of agent. The default is ``"BiResDDPG"``.
            state_shape (iterable of int):
            action_dim (int):
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
        kwargs["name"] = name
        super().__init__(**kwargs)
        self._eta = eta

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors1, td_errors2 = self._compute_td_error_body(
                    states, actions, next_states, rewards, dones)
                critic_loss = tf.reduce_mean(
                    tf.square(td_errors1) * weights +
                    tf.square(td_errors2) * weights * self.discount * self._eta)

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                next_action = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic(states, next_action))

            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            update_target_variables(
                self.critic_target.weights, self.critic.weights, self.tau)
            update_target_variables(
                self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, tf.abs(td_errors1) + tf.abs(td_errors2)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        """
        Compute TD error

        Args:
            states
            actions
            next_states
            rewars
            dones

        Returns:
            np.ndarray: Sum of two TD errors.
        """
        td_error1, td_error2 = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.squeeze(np.abs(td_error1.numpy()) + np.abs(td_error2.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)
            # Compute standard TD error
            target_Q = self.critic_target(next_states, self.actor_target(next_states))
            target_Q = rewards + (not_dones * self.discount * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            current_Q = self.critic(states, actions)
            td_errors1 = target_Q - current_Q
            # Compute residual TD error
            next_actions = tf.stop_gradient(self.actor(next_states))
            target_Q = self.critic(next_states, next_actions)
            target_Q = rewards + (not_dones * self.discount * target_Q)
            current_Q = tf.stop_gradient(self.critic_target(states, actions))
            td_errors2 = target_Q - current_Q
        return td_errors1, td_errors2

    @staticmethod
    def get_argument(parser=None):
        """
        Create or update argument parser for command line program

        Args:
            parser (argparse.ArgParser, optional): argument parser

        Returns:
            argparse.ArgParser: argument parser
        """
        parser = DDPG.get_argument(parser)
        parser.add_argument('--eta', type=float, default=0.05)
        return parser
