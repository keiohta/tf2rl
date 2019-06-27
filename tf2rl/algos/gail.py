import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import Policy
from tf2rl.networks.spectral_norm_dense import SNDense
from tensorflow.keras.layers import Dense


class Discriminator(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[32, 32], enable_sn=False, name="Discriminator"):
        super().__init__(name=name)

        DenseClass = SNDense if enable_sn else Dense
        self.l1 = DenseClass(units[0], name="L1", activation="relu")
        self.l2 = DenseClass(units[1], name="L2", activation="relu")
        self.l3 = DenseClass(1, name="L3", activation="sigmoid")

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        with tf.device("/cpu:0"):
            self([dummy_state, dummy_action])

    def call(self, inputs):
        features = tf.concat(inputs, axis=1)
        features = self.l1(features)
        features = self.l2(features)
        return self.l3(features)


class GAIL(Policy):
    def __init__(
            self,
            state_shape,
            action_dim,
            units=[32, 32],
            lr=0.001,
            enable_sn=False,
            name="GAIL",
            **kwargs):
        super().__init__(name=name, memory_capacity=1, **kwargs)
        self.disc = Discriminator(state_shape, action_dim, units, enable_sn)
        # TODO: Check beta1 of GAIL is same with GAN
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0.5)

    def train(self, agent_states, agent_acts, expert_states, expert_acts):
        loss, accuracy = self._train_body(agent_states, agent_acts,
                                          expert_states, expert_acts)
        tf.summary.scalar(name="DiscriminatorLoss", data=loss)
        tf.summary.scalar(name="Accuracy", data=accuracy)
        # TODO: Summarize kl-divergence and classification accuracy

    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts):
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                real_logit = self.disc([expert_states, expert_acts])
                fake_logit = self.disc([agent_states, expert_acts])
                loss = -(tf.reduce_mean(tf.math.log(real_logit + epsilon)) +
                         tf.reduce_mean(tf.math.log(1. - fake_logit + epsilon)))
                accuracy = \
                    tf.reduce_mean(tf.cast(real_logit > 0.5, tf.float32)) / 2. + \
                    tf.reduce_mean(tf.cast(fake_logit < 0.5, tf.float32)) / 2.
            grads = tape.gradient(loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.disc.trainable_variables))
        return loss, accuracy

    def inference(self, states, actions):
        if states.ndim == actions.ndim == 1:
            states = np.expand_dims(states, axis=0)
            actions = np.expand_dims(actions, axis=0)
        return self._inference_body(states, actions)

    @tf.function
    def _inference_body(self, states, actions):
        with tf.device(self.device):
            return tf.math.log(self.disc([states, actions]) + 1e-8)

    @staticmethod
    def get_argument(parser=None):
        import argparse
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--enable-sn', action='store_true')
        return parser


if __name__ == '__main__':
    # TODO: Solve binary classification problem using GAIL
    pass
