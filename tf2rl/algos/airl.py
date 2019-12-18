import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.networks.spectral_norm_dense import SNDense


class BaseModel(tf.keras.Model):
    def __init__(self, state_shape, action_dim=None, units=[32, 32],
                 enable_sn=False, output_activation="sigmoid",
                 name="Discriminator"):
        super().__init__(self, name=name)

        DenseClass = SNDense if enable_sn else Dense
        self.l1 = DenseClass(units[0], name="L1", activation="relu")
        self.l2 = DenseClass(units[1], name="L2", activation="relu")
        self.l3 = DenseClass(1, name="L3", activation=output_activation)

        dummy_state = tf.constant(
            np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        if action_dim is None:
            dummy_inputs = [dummy_state]
        else:
            dummy_action = tf.constant(
                np.zeros(shape=[1, action_dim], dtype=np.float32))
            dummy_inputs = [dummy_state, dummy_action]
        self._define_shape(dummy_inputs)

    def _define_shape(self, dummy_inputs):
        with tf.device("/cpu:0"):
            self(dummy_inputs)

    def call(self, inputs):
        features = tf.concat(inputs, axis=1)
        features = self.l1(features)
        features = self.l2(features)
        return self.l3(features)


class RewardApproximator(BaseModel):
    def __init__(self, *, name="RewardApprox", **kwargs):
        super().__init__(name=name, **kwargs)


class ValueApproximator(BaseModel):
    def __init__(self, *, name="ValueApproximator", action_dim=None, **kwargs):
        super().__init__(name=name, action_dim=action_dim, **kwargs)

    def _define_shape(self, dummy_inputs):
        with tf.device("/cpu:0"):
            self(dummy_inputs[0])

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        return self.l3(features)


class AIRL(IRLPolicy):
    def __init__(
            self,
            state_shape,
            action_dim=None,
            state_only=True,
            units=[32, 32],
            lr=0.001,
            enable_sn=False,
            name="AIRL",
            **kwargs):
        super().__init__(name=name, n_training=1, **kwargs)
        self._state_only = state_only
        if state_only:
            action_dim = None
        self.rew_net = RewardApproximator(
            state_shape=state_shape, action_dim=action_dim,
            units=units, enable_sn=enable_sn)
        self.val_net = ValueApproximator(
            state_shape=state_shape, units=units, enable_sn=enable_sn)
        self.rew_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr)
        self.val_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr)

    def train(self, agent_states, agent_acts, agent_next_states, agent_logps,
              expert_states, expert_acts, expert_next_states, expert_logps):
        loss = self._train_body(
            agent_states, agent_acts, agent_next_states, agent_logps,
            expert_states, expert_acts, expert_next_states, expert_logps)
        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)

    @tf.function
    def _train_body(self, agent_states, agent_acts, agent_next_states, agent_logps,
                    expert_states, expert_acts, expert_next_states, expert_logps):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self._state_only:
                    real_rews = self.rew_net([expert_states])
                    fake_rews = self.rew_net([agent_states])
                else:
                    real_rews = self.rew_net([expert_states, expert_acts])
                    fake_rews = self.rew_net([agent_states, agent_acts])
                real_vals = self.val_net(expert_states)
                real_next_vals = self.val_net(expert_next_states)
                fake_vals = self.val_net(agent_states)
                fake_next_vals = self.val_net(agent_next_states)
                loss = tf.reduce_mean(
                    fake_rews + self.discount * fake_next_vals - fake_vals - agent_logps)
                loss -= tf.reduce_mean(
                    real_rews + self.discount * real_next_vals - real_vals - expert_logps)
            grads_val = tape.gradient(loss, self.val_net.trainable_variables)
            grads_rew = tape.gradient(loss, self.rew_net.trainable_variables)
            self.val_optimizer.apply_gradients(
                zip(grads_val, self.val_net.trainable_variables))
            self.rew_optimizer.apply_gradients(
                zip(grads_rew, self.rew_net.trainable_variables))

        return loss

    def inference(self, states, actions, **kwargs):
        if states.ndim == actions.ndim == 1:
            states = np.expand_dims(states, axis=0)
            actions = np.expand_dims(actions, axis=0)
        return self._inference_body(states, actions)

    @tf.function
    def _inference_body(self, states, actions):
        with tf.device(self.device):
            if self._state_only:
                return self.rew_net([states])
            else:
                return self.rew_net([states, actions])

    @staticmethod
    def get_argument(parser=None):
        import argparse
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--enable-sn', action='store_true')
        return parser
