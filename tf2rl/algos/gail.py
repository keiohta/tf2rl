import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.networks.spectral_norm_dense import SNDense


class Discriminator(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(32, 32),
                 enable_sn=False, output_activation="sigmoid",
                 name="Discriminator"):
        super().__init__(name=name)

        DenseClass = SNDense if enable_sn else Dense
        self.l1 = DenseClass(units[0], name="L1", activation="relu")
        self.l2 = DenseClass(units[1], name="L2", activation="relu")
        self.l3 = DenseClass(1, name="L3", activation=output_activation)

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=(1, action_dim), dtype=np.float32))
        with tf.device("/cpu:0"):
            self(tf.concat((dummy_state, dummy_action), axis=1))

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        return self.l3(features)

    def compute_reward(self, inputs):
        return tf.math.log(self(inputs) + 1e-8)


class GAIL(IRLPolicy):
    """
    Generative Adversarial Imitation Learning (GAIL) Agent: https://arxiv.org/abs/1606.03476

    Command Line Args:

        * ``--n-warmup`` (int): Number of warmup steps before training. The default is ``1e4``.
        * ``--batch-size`` (int): Batch size of training. The default is ``32``.
        * ``--gpu`` (int): GPU id. ``-1`` disables GPU. The default is ``0``.
        * ``--memory-capacity`` (int): Replay Buffer size. The default is ``1e4``.
        * ``--enable-sn``: Enable Spectral Normalization
    """
    def __init__(
            self,
            state_shape,
            action_dim,
            units=[32, 32],
            lr=0.001,
            enable_sn=False,
            name="GAIL",
            **kwargs):
        """
        Initialize GAIL

        Args:
            state_shape (iterable of int):
            action_dim (int):
            units (iterable of int): The default is ``[32, 32]``
            lr (float): Learning rate. The default is ``0.001``
            enable_sn (bool): Whether enable Spectral Normalization. The defailt is ``False``
            name (str): The default is ``"GAIL"``
        """
        super().__init__(name=name, n_training=1, **kwargs)
        self.disc = Discriminator(
            state_shape=state_shape, action_dim=action_dim,
            units=units, enable_sn=enable_sn)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0.5)

    def train(self, agent_states, agent_acts,
              expert_states, expert_acts, **kwargs):
        """
        Train GAIL

        Args:
            agent_states
            agent_acts
            expert_states
            expected_acts
        """
        loss, accuracy, js_divergence = self._train_body(
            agent_states, agent_acts, expert_states, expert_acts)
        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)

    def _compute_js_divergence(self, fake_logits, real_logits):
        m = (fake_logits + real_logits) / 2.
        return tf.reduce_mean((
            fake_logits * tf.math.log(fake_logits / m + 1e-8) + real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.)

    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts):
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                real_logits = self.disc(tf.concat((expert_states, expert_acts), axis=1))
                fake_logits = self.disc(tf.concat((agent_states, agent_acts), axis=1))
                loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                         tf.reduce_mean(tf.math.log(1. - fake_logits + epsilon)))
            grads = tape.gradient(loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.disc.trainable_variables))

        accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                    tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
        js_divergence = self._compute_js_divergence(
            fake_logits, real_logits)
        return loss, accuracy, js_divergence

    def inference(self, states, actions, next_states):
        """
        Infer Reward with GAIL

        Args:
            states
            actions
            next_states

        Returns:
            tf.Tensor: Reward
        """
        if states.ndim == actions.ndim == 1:
            states = np.expand_dims(states, axis=0)
            actions = np.expand_dims(actions, axis=0)
        inputs = np.concatenate((states, actions), axis=1)
        return self._inference_body(inputs)

    @tf.function
    def _inference_body(self, inputs):
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        parser.add_argument('--enable-sn', action='store_true')
        return parser
