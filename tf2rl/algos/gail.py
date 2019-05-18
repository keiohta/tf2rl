import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import Policy


class Discriminator(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[32, 32], name="Discriminator"):
        super().__init__(name=name)

        self.l1 = tf.keras.layers.Dense(units[0], name="L1", activation="relu")
        self.l2 = tf.keras.layers.Dense(units[1], name="L2", activation="relu")
        self.l3 = tf.keras.layers.Dense(1, name="L3", activation="sigmoid")

        dummy_state = tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float64))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float64))
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
            name="GAIL",
            **kwargs):
        super().__init__(name=name, memory_capacity=1, **kwargs)
        self.disc = Discriminator(state_shape, action_dim, units)
        # TODO: Check beta1 of GAIL is same with GAN
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=lr, beta1=0.5)

    def train(self, agent_states, agent_acts, expert_states, expert_acts):
        loss = self._train_body(agent_states, agent_acts,
                                expert_states, expert_acts)
        tf.contrib.summary.scalar(
            name="DiscriminatorLoss", tensor=loss, family="loss")
        # TODO: Summarize kl-divergence and classification accuracy

    # @tf.contrib.eager.defun
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts):
        epsilon = 1e-8
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                true_logit = self.disc([expert_states, expert_acts])
                fake_logit = self.disc([agent_states, expert_acts])
                loss = -(tf.reduce_mean(tf.log(true_logit + epsilon)) + \
                         tf.reduce_mean(tf.log(1. - fake_logit + epsilon)))
            grads = tape.gradient(loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.disc.trainable_variables))
        return loss

    @tf.contrib.eager.defun
    def inference(self, states, actions):
        with tf.device(self.device):
            return tf.log(self.disc([states, actions]) + 1e-8)


if __name__ == '__main__':
    # TODO: Solve binary classification problem using GAIL
    pass
