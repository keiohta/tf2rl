import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class StateNetwork(tf.keras.Model):
    def __init__(self,units,activation,name):
        super().__init__(name=name)

        self.L = []
        for i in range(len(units)):
            self.L.append(Dense(units[i],
                                activation=activation[i],
                                name=f"L{i}"))

    def _dummy_call(self,state_shape,action_dim):
        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros((1,)+state_shape, dtype=np.float32)))

    def call(self, inputs):
        for l in self.L:
            inputs = l(inputs)
        return inputs

class StateActionNetwork(StateNetwork):
    def _dummy_call(self,state_shape,action_dim):
        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros((1,)+state_shape, dtype=np.float32)),
                 tf.constant(np.zeros([1, action_dim], dtype=np.float32)))

    def call(self, inputs):
        states, action = inputs
        return super().call(tf.concat([states, action], axis=1))
