import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tf2rl.algos.sac import SAC
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.networks.dmc_model import Encoder, Decoder
from tf2rl.tools.img_tools import center_crop, preprocess_img


class SACAE(SAC):
    """
    SAC+AE Agent: https://arxiv.org/abs/1910.01741

    Command Line Args:

        * ``--n-warmup`` (int): Number of warmup steps before training. The default is ``1e4``.
        * ``--batch-size`` (int): Batch size of training. The default is ``32``.
        * ``--gpu`` (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        * ``--memory-capacity`` (int): Replay Buffer size. The default is ``1e5``.
        * ``--alpha`` (float): Temperature parameter. The default is ``0.2``.
        * ``--auto-alpha``: Automatic alpha tuning.
        * ``--stop-q-grad``: Whether stop gradient after convolutional layers at Encoder
    """
    def __init__(self,
                 action_dim,
                 obs_shape=(84, 84, 9),
                 n_conv_layers=4,
                 n_conv_filters=32,
                 feature_dim=50,
                 tau_encoder=0.05,
                 tau_critic=0.01,
                 auto_alpha=True,
                 lr_sac=1e-3,
                 lr_encoder=1e-3,
                 lr_decoder=1e-3,
                 update_critic_target_freq=2,
                 update_actor_freq=2,
                 lr_alpha=1e-4,
                 init_temperature=0.1,
                 stop_q_grad=False,
                 lambda_latent_val=1e-06,
                 decoder_weight_lambda=1e-07,
                 skip_making_decoder=False,
                 name="SACAE",
                 **kwargs):
        """
        Initialize SAC+AE

        Args:
            action_dim (int):
            obs_shape: (iterable of int): The default is ``(84, 84, 9)``
            n_conv_layers (int): Number of convolutional layers at encoder. The default is ``4``
            n_conv_filters (int): Number of filters in convolutional layers. The default is ``32``
            feature_dim (int): Number of features after encoder. This features are treated as SAC input. The default is ``50``
            tau_encoder (float): Target network update rate for Encoder. The default is ``0.05``
            tau_critic (float): Target network update rate for Critic. The default is ``0.01``
            auto_alpha (bool): Automatic alpha tuning. The default is ``True``
            lr_sac (float): Learning rate for SAC. The default is ``1e-3``
            lr_encoder (float): Learning rate for Encoder. The default is ``1e-3``
            lr_decoder (float): Learning rate for Decoder. The default is ``1e-3``
            update_critic_target_freq (int): The default is ``2``
            update_actor_freq (int): The default is ``2``
            lr_alpha (alpha): Learning rate for alpha. The default is ``1e-4``.
            init_temperature (float): Initial temperature. The default is ``0.1``
            stop_q_grad (bool): Whether sotp gradient propagation after encoder convolutional network. The default is ``False``
            lambda_latent_val (float): AE loss = REC loss + ``lambda_latent_val`` * latent loss. The default is ``1e-6``
            decoder_weight_lambda (float): Weight decay of AdamW for Decoder. The default is ``1e-7``
            skip_making_decoder (bool): Whther skip making Decoder. The default is ``False``
            name (str): Name of network. The default is ``"SACAE"``
            max_action (float):
            actor_units (iterable of int): Numbers of units at hidden layers of actor. The default is ``(256, 256)``.
            critic_units (iterable of int): Numbers of units at hidden layers of critic. The default is ``(256, 256)``.
            alpha (float): Temperature parameter. The default is ``0.2``.
            n_warmup (int): Number of warmup steps before training. The default is ``int(1e4)``.
            memory_capacity (int): Replay Buffer size. The default is ``int(1e6)``.
            batch_size (int): Batch size. The default is ``256``.
            discount (float): Discount factor. The default is ``0.99``.
            max_grad (float): Maximum gradient. The default is ``10``.
            gpu (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        """
        super().__init__(state_shape=(feature_dim,),
                         action_dim=action_dim,
                         name=name,
                         lr=lr_sac,
                         lr_alpha=lr_alpha,
                         tau=tau_critic,
                         auto_alpha=auto_alpha,
                         init_temperature=init_temperature,
                         **kwargs)
        self._encoder = Encoder(obs_shape=obs_shape,
                                feature_dim=feature_dim,
                                n_conv_layers=n_conv_layers,
                                n_conv_filters=n_conv_filters,
                                name="encoder")
        self._encoder_target = Encoder(obs_shape=obs_shape,
                                       feature_dim=feature_dim,
                                       n_conv_layers=n_conv_layers,
                                       n_conv_filters=n_conv_filters,
                                       name="encoder_target")
        update_target_variables(self._encoder_target.weights, self._encoder.weights, tau=1.)

        self._encoder_optimizer = tf.keras.optimizers.Adam(lr=lr_encoder)
        if not skip_making_decoder:
            self._decoder = Decoder()
            self._lambda_latent_val = lambda_latent_val
            self._decoder_optimizer = tfa.optimizers.AdamW(learning_rate=lr_decoder, weight_decay=decoder_weight_lambda)

        self._stop_q_grad = stop_q_grad
        self._input_img_size = obs_shape[0]
        self._tau_encoder = tau_encoder
        self._n_update = 0
        self._update_critic_target_freq = update_critic_target_freq
        self._update_actor_freq = update_actor_freq
        self._feature_dim = feature_dim
        self.state_ndim = 3

    def get_action(self, state, test=False):
        """
        Get action

        Args:
            state: Observation state
            test (bool): When ``False`` (default), policy returns exploratory action.

        Returns:
            tf.Tensor or float: Selected action

        Notes:
            When the input image have different size, cropped image is used
        """
        if state.shape[:-3] != self._input_img_size:
            state = center_crop(state, self._input_img_size)
        return super().get_action(state, test)

    @tf.function
    def _get_action_body(self, state, test):
        encoded_state = self._encoder(state)
        actions, log_pis = self.actor(encoded_state, test)
        return actions

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        """
        Train SAC+AE

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

        # Update critic
        td_errors, qf_loss = self._update_critic(
            states, actions, next_states, rewards, dones, weights)
        tf.summary.scalar(name=self.policy_name + "/critic_loss", data=qf_loss)
        if self._n_update % self._update_critic_target_freq == 0:
            update_target_variables(self.qf1_target.weights, self.qf1.weights, self.tau)
            update_target_variables(self.qf2_target.weights, self.qf2.weights, self.tau)

        # Update actor
        if self._n_update % self._update_actor_freq == 0:
            obs_features = self._encoder(states)
            actor_loss, logp_min, logp_max, logp_mean, alpha_loss = self._update_actor(obs_features)
            tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
            tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
            tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
            tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
            if self.auto_alpha:
                tf.summary.scalar(name=self.policy_name + "/log_ent", data=self.log_alpha)
                tf.summary.scalar(name=self.policy_name + "/logp_mean+target", data=logp_mean + self.target_alpha)
            tf.summary.scalar(name=self.policy_name + "/ent", data=self.alpha)
            tf.summary.scalar(name=self.policy_name + "/alpha_loss", data=alpha_loss)

        # Update encoder/decoder
        rec_loss, latent_loss = self._update_encoder(states)
        tf.summary.scalar(name=self.policy_name + "/rec_loss", data=rec_loss)
        tf.summary.scalar(name=self.policy_name + "/latent_loss", data=latent_loss)

        self._n_update += 1

        return qf_loss

    @tf.function
    def _update_critic(self, obses, actions, next_obses, rewards, dones, weights):
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2
            rewards = tf.squeeze(rewards, axis=1)
            dones = tf.squeeze(dones, axis=1)

            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                next_obs_features = self._encoder(next_obses)
                next_actions, next_logps = self.actor(next_obs_features)

                next_obs_features_target = self._encoder_target(next_obses)
                next_target_q1 = tf.stop_gradient(self.qf1_target(next_obs_features_target, next_actions))
                next_target_q2 = tf.stop_gradient(self.qf2_target(next_obs_features_target, next_actions))
                min_next_target_q = tf.minimum(next_target_q1, next_target_q2)

                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * (min_next_target_q - self.alpha * next_logps))

                obs_features = self._encoder(obses, stop_q_grad=self._stop_q_grad)
                current_q1 = self.qf1(obs_features, actions)
                current_q2 = self.qf2(obs_features, actions)
                td_loss_q1 = tf.reduce_mean((target_q - current_q1) ** 2)
                td_loss_q2 = tf.reduce_mean((target_q - current_q2) ** 2)  # Eq.(6)

            q1_grad = tape.gradient(td_loss_q1, self._encoder.trainable_variables + self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self._encoder.trainable_variables + self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss_q2, self._encoder.trainable_variables + self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self._encoder.trainable_variables + self.qf2.trainable_variables))

        return td_loss_q1 + td_loss_q2, td_loss_q1

    @tf.function
    def _update_encoder(self, obses):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                obs_features = self._encoder(obses, stop_q_grad=self._stop_q_grad)

                # Compute loss of AE
                rec_obses = self._decoder(obs_features)
                true_obses = preprocess_img(obses)
                rec_loss = tf.reduce_mean(tf.keras.losses.MSE(true_obses, rec_obses))
                latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.math.pow(obs_features, 2), axis=1))
                ae_loss = rec_loss + self._lambda_latent_val * latent_loss

            encoder_grads = tape.gradient(ae_loss, self._encoder.trainable_variables)
            self._encoder_optimizer.apply_gradients(zip(encoder_grads, self._encoder.trainable_variables))
            decoder_grads = tape.gradient(ae_loss, self._decoder.trainable_variables)
            self._encoder_optimizer.apply_gradients(zip(decoder_grads, self._decoder.trainable_variables))
            update_target_variables(
                self._encoder_target.weights, self._encoder.weights, self._tau_encoder)

        return rec_loss, latent_loss

    @staticmethod
    def get_argument(parser=None):
        """
        Create or update argument parser for command line program

        Args:
            parser (argparse.ArgParser, optional): argument parser

        Returns:
            argparse.ArgParser: argument parser
        """
        parser = SAC.get_argument(parser)
        parser.add_argument('--stop-q-grad', action="store_true")
        parser.add_argument('--memory-capacity', type=int, default=int(1e5))
        return parser

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        return 0.
